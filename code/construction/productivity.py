from utils.get_entries import get_entries
from construction.check import check_known, delete_comp
from construction.entry import Entry
from random import shuffle
from copy import deepcopy


def coefficient(a, b):
    r, sa, sb = 0, sum([a[x] for x in a]), sum([b[x] for x in b])
    for x in a:
        r += ((a[x] / sa) ** 0.5) * ((b[x] / sb if x in b else 0) ** 0.5)
    return 1 - r


def deal_pro(con):
    print("Construction for Productivity")
    root, pro_type, max_seen_len = con.corp_folder, con.pro, con.pro_len
    mr = (con.corp == "e2e")
    train, test = get_entries(root, con.corp)

    # To avoid inconsistent domain distributions of training sets, we only use 4 domains in WebNLG
    test_cate = {'Monument', 'Astronaut', 'University', 'Company'}

    def check_cate(e):
        return e.category in test_cate or con.corp == "e2e"

    saved_test = [e for e in test if len(e.modifiedtripleset.triples) > max_seen_len and check_cate(e)]

    cnt = { }

    # Invisible
    saved_unseen, inc_u = [ ], [0] * 10
    unseen_candi = [ ]
    for e in train:
        if not check_cate(e):
            continue
        n = len(e.modifiedtripleset.triples)
        if n <= max_seen_len:
            saved_unseen.append(e)
            inc_u[n - 1] += 1
            unseen_candi.append(Entry(e, mr))
            for x in Entry(e, mr).triples:
                if x not in cnt:
                    cnt[x] = 0
                cnt[x] += 1

    # Visible
    saved_seen, inc_s = [ ], [0] * 10
    candi = [ ]
    for e in train:
        if not check_cate(e):
            continue
        n = len(e.modifiedtripleset.triples)
        if n > max_seen_len:
            candi.append(Entry(e, mr))

    for x in candi:
        for y in x.triples:
            if y not in cnt:
                cnt[y] = 0
    unseen_cnt = deepcopy(cnt)

    def add(x, val):
        for y in x.triples:
            cnt[y] += val

    # When the dataset is large enough,
    # the requirement for divergence can be easily satisfied by constructing Visible by random selection
    # Control is applied if the original dataset is not large enough
    control_type = "no_control" if con.corp in ["e2e"] else "control"

    while len(candi) != 0:
        if control_type == "no_control":
            candi.sort(key=lambda x: len(x.triples))
        else:
            candi.sort(key=lambda x: (sum([0] + [unseen_cnt[y] - cnt[y] for y in x.triples])))
        x = candi.pop( )
        add(x, 1)
        num = len(x.triples)
        unseen_candi.sort(key=lambda x: (sum([0] + [cnt[y] - unseen_cnt[y] for y in x.triples])))
        remove_idx = [ ]
        for i in range(len(unseen_candi) - 1, -1, -1):
            if len(unseen_candi[i].triples) <= num:
                add(unseen_candi[i], -1)
                remove_idx.append(i)
                num -= len(unseen_candi[i].triples)
        c = coefficient(unseen_cnt, cnt)
        if num == 0 and c <= con.pro_co_limit:
            saved_seen.append(x.data)
            for i in remove_idx:
                unseen_candi.pop(i)
        else:
            add(x, -1)
            for i in remove_idx:
                add(unseen_candi[i], 1)

    saved_seen = saved_seen + [x.data for x in unseen_candi]
    for x in saved_seen:
        inc_s[len(x.modifiedtripleset.triples) - 1] += 1

    print("Divergence (all units):", coefficient(unseen_cnt, cnt))
    atoms = set( )
    for e in saved_test:
        for x in Entry(e, mr).triples:
            atoms.add(x)
    unseen_cnt = {x: unseen_cnt[x] for x in atoms}
    cnt = {x: cnt[x] for x in atoms}
    print("Divergence (atoms only):", coefficient(unseen_cnt, cnt))

    print("Length distribution / #Samples / #Units (Invisible):")
    print(inc_u, sum(inc_u), sum([(i + 1) * x for i, x in enumerate(inc_u)]))
    print("Length distribution / #Samples / #Units (Visible):")
    print(inc_s, sum(inc_s), sum([(i + 1) * x for i, x in enumerate(inc_s)]))

    if pro_type == "visible":
        saved_train = saved_seen
    elif pro_type == "invisible":
        saved_train = saved_unseen

    # check_known(saved_train, [saved_test])
    saved_test = delete_comp(saved_unseen, saved_test, mr)
    saved_test = delete_comp(saved_seen, saved_test, mr)
    # check_known(saved_train, [saved_test])

    inc_t = [0] * 10
    for e in saved_test:
        inc_t[len(e.modifiedtripleset.triples) - 1] += 1
    print("Length distribution / #Samples / #Units (Test):")
    print(inc_t, sum(inc_t), sum([(i + 1) * x for i, x in enumerate(inc_t)]))

    return (saved_train, saved_test), (saved_test,)
