from utils.get_entries import get_entries
from random import shuffle
from copy import deepcopy
from utils.data_match import rearrange, get_matched_samples, get_all_plans
from utils.convert import bool_convert


def check(arr, n):
    for i in range(n):
        if arr[i] != i:
            return True
    return False


def get_pairs(e, corp):
    if corp == "webnlg":
        return [(x.s, x.o) for x in e.modifiedtripleset.triples]
    elif corp == "e2e":
        return [bool_convert(x.p, x.o) for x in e.modifiedtripleset.triples]


def deal_ord(con):
    print("Construction for Order invariance")
    root, loc_train, loc_test = con.corp_folder, con.ord, con.ord_aspect
    train, test = get_entries(root, con.corp)
    if loc_train == "original":
        saved_train = train
        print("#Samples (Original): ", len(saved_train))
    else:
        saved_train, suc, tot = [ ], 0, 0
        for e in train:
            samples, sc, inc = get_matched_samples(e, get_pairs(e, con.corp))
            saved_train += samples
            suc += sc
            tot += inc
        print("#Samples (Match): ", len(saved_train))
        print("Matching success rate of data units (Match): %.4f" % (suc / tot))
    if loc_test == "performance":
        print("#Samples (Test-perf): ", len(test))
        return (saved_train, test), (test,)
    elif loc_test == "cwio":
        saved_test = [ ]
        saved_pairs = [ ]
        for e in test:
            n = len(e.modifiedtripleset.triples)
            if n == 1:
                continue
            tri = e.modifiedtripleset.triples
            saved_pairs.append([(deepcopy(x.s), deepcopy(x.o)) for x in tri])
            saved_test.append(e)
        print("#Samples (Test-cwio):", len(saved_test))
        return (saved_train, test), ((saved_test,), saved_pairs)
    else:
        saved_test = [ ]
        saved_plans = [ ]
        saved_pairs = [ ]
        original_test = [ ]
        for e in test:
            n = len(e.modifiedtripleset.triples)
            if n == 1:
                continue
            pairs = get_pairs(e, con.corp)
            p = get_all_plans(e, pairs)
            if p != None:
                original_test.append(e)
                de = deepcopy(e)
                tri = de.modifiedtripleset.triples
                saved_pairs.append(pairs)
                arr = [_ for _ in range(n)]
                shuffle(arr)
                de.modifiedtripleset.triples = rearrange(tri, arr)
                saved_test.append(de)
                saved_plans.append(p)
        print("#Samples (Test-main):", len(saved_test))
        another = deepcopy(saved_test)
        for e in another:
            tri = e.modifiedtripleset.triples
            n = len(tri)
            arr = [_ for _ in range(n)]
            shuffle(arr)
            while not check(arr, n):
                shuffle(arr)
            e.modifiedtripleset.triples = rearrange(tri, arr)
        return (saved_train, test), ((another, saved_test), saved_plans, saved_pairs, original_test)