def ordered(x, y):
    if x > y:
        return y, x
    else:
        return x, y


def check_known(train, candi):
    seen = set( )
    for e in train:
        for x in e.modifiedtripleset.triples:
            t = (x.s, x.p, x.o)
            seen.add(t)
    print("Known Triple Rate")
    for corp in candi:
        inc_all, inc_in = 0, 0
        all = set( )
        for e in corp:
            for x in e.modifiedtripleset.triples:
                t = (x.s, x.p, x.o)
                all.add(t)
        for x in all:
            inc_all += 1
            if x in seen:
                inc_in += 1
        print(inc_in, inc_all, inc_in / inc_all)


def delete_comp(train, test, mr):
    seen = set( )
    for e in train:
        for x in e.modifiedtripleset.triples:
            if not mr:
                t = (x.s, x.p, x.o)
            else:
                t = (x.p, x.o)
            seen.add(t)
    saved_test = [ ]
    cnt = 0
    for e in test:
        flag = True
        for x in e.modifiedtripleset.triples:
            if not mr:
                t = (x.s, x.p, x.o)
            else:
                t = (x.p, x.o)
            if t not in seen:
                flag = False
                break
        if flag:
            saved_test.append(e)
        cnt += 1
    return saved_test
