from utils.get_entries import get_entries
from copy import deepcopy
from random import shuffle, choice
from corpusreader.benchmark_reader_e2e import Entry


def clean(s):
    return s.replace('_', ' ')


def has_digit(s):
    for x in s:
        if x.isdigit( ):
            return True
    return False


def check_overlap(x, s):
    for y in s:
        if x in y:
            return True
    return False


def deal_rla_webnlg(con):
    print("Construction for Rule learnability")
    root, unk_rep = con.corp_folder, con.ent_unk
    train, test = get_entries(root, con.corp)
    saved_test, saved_replaced = [ ], [ ]

    for e in test:
        # Determine the entities to be replaced
        tri = e.modifiedtripleset.triples
        s = [clean(x.s) for x in tri]
        o = [clean(x.o) for x in tri]
        all_entities = set(s + o)
        s_entities = set(s)
        text = [t.lex for t in e.lexs]
        replaced, cnt = { }, 0
        for x in sorted(list(s_entities)):
            if has_digit(x):
                continue
            overlap = False
            for y in all_entities:
                if x != y and x.lower( ) in y.lower( ):
                    overlap = True
                    break
            if overlap:
                continue
            appear = True
            for t in text:
                if t.lower( ).find(x.lower( )) == -1:
                    appear = False
                    break
            if not appear:
                continue
            cnt += 1
            replaced[x] = unk_rep + " " + str(cnt)

        if cnt == 0:
            continue

        # Exclude entities that have exactly the same pattern as other entities in the sample
        saved_tri = [ ]
        record = set( )
        for x in e.modifiedtripleset.triples:
            s, p, o = clean(x.s), x.p, clean(x.o)
            if s not in replaced and o not in replaced:
                record.add((s, p))
                record.add((p, o))
        for x in e.modifiedtripleset.triples:
            s, p, o = clean(x.s), x.p, clean(x.o)
            if s in replaced and (p, o) in record:
                replaced.pop(s)
            if o in replaced and (s, p) in record:
                replaced.pop(o)

        if len(replaced) == 0:
            continue

        # Perform Replacement
        for x in e.modifiedtripleset.triples:
            y = deepcopy(x)
            s, p, o = clean(y.s), y.p, clean(y.o)
            if s in replaced:
                y.s = replaced[s]
            if o in replaced:
                y.o = replaced[o]
            saved_tri.append(y)
        e.modifiedtripleset.triples = saved_tri
        saved_test.append(e)
        saved_replaced.append(replaced)

    print("#Samples (Train):", len(train))
    print("#Samples (Test):", len(saved_test))
    return (train, test), ((saved_test, ), saved_replaced)


def deal_rla_e2e(con):
    print("Construction for Rule learnability")
    root = con.corp_folder
    train, test = get_entries(root, con.corp)
    possible_val = {"name": set()}
    for x in train:
        for y in x.modifiedtripleset.triples:
            if y.p not in possible_val.keys( ):
                possible_val[y.p] = set( )
            possible_val[y.p].add(y.o)
            possible_val["name"].add(y.s)

    cond = ["eat type", "food", "price range", "customer rating", "area", "family friendly"]
    num_related = {"customer rating": ["? out of 5"], "price range": ["less than ?", "more than ?"]}

    cnt = { }
    for x in cond:
        for val in possible_val[x]:
            cnt[val] = cnt[val] + 1 if val in cnt.keys( ) else 1
    candi_vals = {x: [val for val in y if cnt[val] == 1] for x, y in possible_val.items( ) if x in cond}
    un_cond = [x for x in possible_val if x not in cond]
    for x in un_cond:
        p = [ ]
        for val in possible_val[x]:
            flag = True
            for y in cond:
                for val_y in possible_val[y]:
                    if val_y in val:
                        flag = False
            if flag:
                p.append(val)
        if x != "name":
            p.append("none")
        candi_vals[x] = p

    candi_vals = {x: sorted(y) for x, y in candi_vals.items( )}
    for x in num_related:
        candi_vals[x] = num_related[x]

    saved_test, saved_choice = [ ], [ ]

    def dfs(x, cur):
        if x == len(cond):
            if len(cur) != 0:
                cnt, val_cur = 0, [ ]
                for i in range(len(cur)):
                    if '?' in cur[i][1]:
                        cur[i] = (cur[i][0], cur[i][1].replace('?', f"{con.val_unk} {chr(ord('A') + cnt)}"))
                        val_cur.append(cur[i])
                        cnt += 1
                if cnt == 0:
                    return
                for r in un_cond:
                    v = choice(candi_vals[r])
                    if v != "none":
                        cur += [(r, v)]
                # print(cur)
                shuffle(cur)
                e = Entry( )
                e.modifiedtripleset.fill_tripleset_from_kv(cur)
                saved_test.append(e)
                saved_choice.append(val_cur)
            return
        for y in candi_vals[cond[x]]:
            dfs(x + 1, cur + [(cond[x], y)])
        dfs(x + 1, cur)

    # Enumerate all possible replacements
    dfs(0, [ ])

    print("#Samples (Train):", len(train))
    print("#Samples (Test):", len(saved_test))
    return (train, test), ((saved_test, ), saved_choice, {x: possible_val[x] for x in num_related})


def deal_rla(con):
    if con.corp == "webnlg":
        return deal_rla_webnlg(con)
    elif con.corp == "e2e":
        return deal_rla_e2e(con)