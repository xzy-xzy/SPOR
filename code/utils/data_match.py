from Levenshtein import distance as editdis
from copy import deepcopy
import numpy as np
from utils.convert import so_conv


def kendall(o1, o2):
    assert len(o1) == len(o2)
    n = len(o1)
    pos = [0] * n
    for i in range(n):
        pos[o2[i]] = i
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[o1[i]] < pos[o1[j]]:
                k += 1
            else:
                k -= 1
    return k / (n * (n - 1) / 2)


mch = [ ]
mn_v = 1e9
synonyms = {
    "united states": ["u s", "usa", "america", "american"],
    "united kingdom": ["uk", "u k"],
    "deceased": ["dead", "passed away"],
    "hardcover": ["hardback"],
    "italy": ["italian"],
    "chinese cuisine": ["china cuisine"],
    "strawberry": ["strawberries"],
    "rock  geology": ["stone"],
    "turkey": ["turkish"],
    "spain": ["spanish"],
}


def most_close(arr, aim):
    ans, mn = 0, 1e9
    for x in arr:
        v = abs(x - aim)
        if v < mn:
            mn = v
            ans = x
    return ans


def dfs(pos, p, n, sch, ft, must_up):
    global mch, mn_v
    if p == n:
        v = np.var(sch)
        if v < mn_v:
            mn_v = v
            mch = [deepcopy(sch)]
        elif v == mn_v:
            mch.append(deepcopy(sch))
        return
    for x in pos[p]:
        if x <= ft and must_up:
            continue
        sch[p] = x
        dfs(pos, p + 1, n, sch, x, must_up)


def lex_match(s, t, strict):
    # To find the set of candidate locations of entity s in text t
    # strict = True: Return an empty list if there is a token that cannot be matched

    global mch, mn_v

    # Find the set of candidate locations for tokens
    L = 2
    s = s.strip( ).split( )
    n, m = len(s), len(t)
    pos = [ ]
    flag = True
    for j in range(n):
        opt = [ ]
        K = len(s[j]) / 2
        min_dist = 1000000000
        for i in range(m):
            min_dist = min(min_dist, editdis(s[j], t[i]))
        if min_dist <= min(K, L):
            for i in range(m):
                if editdis(s[j], t[i]) == min_dist:
                    opt.append(i)
        if opt:
            pos.append(opt)
        else:
            flag = False
    if not flag:
        if strict:
            return [ ]
        else:
            for x in s:
                if len(x) != 1 or x.isdigit( ):
                    flag = True
                    break
            if not flag:
                return [ ]

    # Determine the set of candidate locations that minimize variance
    N = len(pos)
    sch = [0] * N
    mch = [ ]
    mn_v = 1e9
    dfs(pos, 0, N, sch, -1, N >= 10)
    return mch


def get_plan(data, text, D, strict=False):
    S = [x[0] for x in data]
    O = [x[1] for x in data]
    T = text.strip( ).split( )
    pos = { }

    def find_pos(x):
        if x != "none" and x not in pos.keys( ):
            pos[x] = lex_match(x, T, strict)
            if not pos[x] and x in synonyms.keys( ):
                for r in synonyms[x]:
                    pos[x] = lex_match(r, T, strict)
                    if pos[x]:
                        break

    # Find the set of candidate locations for entities
    for s, o in zip(S, O):
        find_pos(s)
        find_pos(o)

    # Locate the entities
    tmp = [(k, v) for k, v in pos.items( )]
    tmp.sort(key=lambda x: len(x[1]))
    occ = set( )
    segocc = set( )
    match = 0
    for k, v in tmp:
        candi = [ ]
        final = None
        for p in v:
            minp = p[0]
            if minp not in occ:
                candi.append(p)
        for p in candi:
            minp = p[0]
            if minp not in segocc:
                final = p
                break
        if (not final) and candi:
            final = candi[0]
        if final:
            minp = final[0]
            pos[k] = minp
            match += 1
            occ.add(minp)
            for x in final:
                segocc.add(x)

    # Determine the order of data units
    plan, id = [ ], 0
    pos["none"] = [ ]
    for s, o in zip(S, O):
        ps, po = pos[s], pos[o]
        if type(ps) == list:
            ps = 1e9
        if type(po) == list:
            po = 1e9
        inner = ps > po
        if s == "none":
            outer = po
        elif D[s] < D[o]:
            outer = ps
        elif D[s] > D[o]:
            outer = po
        else:
            outer = max(ps, po)
        plan.append((id, inner, outer))
        id += 1
    plan.sort(key=lambda x: x[2])

    return plan


def rearrange(tri, arr):
    saved = [ ]
    for x in arr:
        saved.append(tri[x])
    return saved


def calculate_degree(pairs):
    so_pair = [(so_conv(x[0]), so_conv(x[1])) for x in pairs]
    d = { }
    for x, y in so_pair:
        d[x] = d[x] + 1 if x in d.keys( ) else 1
        d[y] = d[y] + 1 if y in d.keys( ) else 1
    return so_pair, d


def get_matched_samples(e, pairs):
    tri = e.modifiedtripleset.triples
    so_pair, d = calculate_degree(pairs)
    samples, suc, tot = [ ], 0, 0
    for t in e.lexs:
        plan = get_plan(so_pair, so_conv(t.lex), d)
        tot += len(plan)
        suc += len([x for x in plan if x[2] != 1e9])
        plan = [x[0] for x in plan]
        r = deepcopy(e)
        r.lexs = [t]
        r.modifiedtripleset.triples = rearrange(tri, plan)
        samples.append(r)
    return samples, suc, tot


def get_all_plans(e, pairs):
    so_pair, d = calculate_degree(pairs)
    plans = [ ]
    for t in e.lexs:
        plan = get_plan(so_pair, so_conv(t.lex), d)
        for x in plan:
            if x[2] == 1e9:
                return None
        plan = [x[0] for x in plan]
        plans.append(plan)
    return plans


def get_kendall(output, pairs):
    n = len(pairs)
    so_pair, d = calculate_degree(pairs)
    plan = get_plan(so_pair, so_conv(output), d)
    occur = len([x for x in plan if x[2] != 1e9])
    plan = [x[0] for x in plan]
    in_plan = [_ for _ in range(n)]
    return n, kendall(plan, in_plan), occur


def get_oc(output, plans, pairs, strict):
    n = len(pairs)
    so_pair, d = calculate_degree(pairs)
    plan = get_plan(so_pair, so_conv(output), d, strict)
    ext = (len([x for x in plan if x[2] != 1e9]) == n)

    if not plans:
        return ext

    original_plan = plan
    plan = [x[0] for x in plan]
    ord_kt = False
    for p in plans:
        if kendall(plan, p) > 0:
            ord_kt = True
            break
    ord_em = False
    for p in plans:
        if tuple(plan) == tuple(p):
            ord_em = True
            break

    return ext, (ord_em, ord_kt), original_plan

