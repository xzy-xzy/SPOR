from utils.convert import convert
from construction.systematicity import deal_sys
from construction.productivity import deal_pro
from construction.order_invariance import deal_ord
from construction.rule_learnability import deal_rla
from pickle import load, dump
from os.path import exists
from os import makedirs
from utils.save_json import save_train_json, save_test_json

def deal_corp(con, prefix, corp):
    res = [ ]
    for e in corp:
        triple = [(convert(x.s), convert(x.p), convert(x.o)) for x in e.modifiedtripleset.triples]
        text = [t.lex for t in e.lexs]
        data = [ ]
        if con.corp == "webnlg":
            for s, p, o in triple:
                data.append("<head>")
                data.append(s)
                data.append("<relation>")
                data.append(p)
                data.append("<tail>")
                data.append(o)
            data = prefix + " ".join(data)
            triple = [x for x in triple]
        elif con.corp == "e2e":
            data.append("name[%s]" % triple[0][0])
            for s, p, o in triple:
                data.append("%s[%s]" % (p, o))
            data = prefix + " ".join(data)
        res.append((data, triple, text))
    return res


def get_corpus(con):
    train_root = con.dataset_folder + "train/" + con.train_name + "/"
    test_root = con.dataset_folder + "test/" + con.test_name + "/"
    train, test, f_train, f_test = None, None, False, False
    for root in [train_root, test_root]:
        if not exists(root):
            makedirs(root)
    train_name = train_root + "dataset.pkl"
    test_name = test_root + "dataset.pkl"

    if not con.check_dataset:
        if exists(train_name):
            train = load(open(train_name, "rb"))
            print("Train Data Found")
            f_train = True
        if exists(test_name):
            test = load(open(test_name, "rb"))
            print("Test Data Found")
            f_test = True

    if (not f_train) or (not f_test):
        if con.aspect == "systematicity":
            res = deal_sys(con)
        elif con.aspect == "productivity":
            res = deal_pro(con)
        elif con.aspect == "order-invariance":
            res = deal_ord(con)
        elif con.aspect == "rule-learnability":
            res = deal_rla(con)
        if con.check_dataset:
            return
        if not f_train:
            train = res[0]
            dump(train, open(train_name, "wb"))
            print("Train Data Saved")
        if not f_test:
            test = res[1]
            dump(test, open(test_name, "wb"))
            print("Test Data Saved")

    save_train_json(train_root, train[0], con)
    print("Train Json Saved")
    save_test_json(test_root, test, con)
    print("Test Json Saved")

    if con.aspect == "order-invariance":
        if con.ord_aspect == "main":
            con.test_plans = test[1]
            con.test_pairs = test[2]
            test = test[0]
        if con.ord_aspect == "cwio":
            con.test_pairs = test[1]
            test = test[0]

    if con.aspect == "rule-learnability":
        if con.corp == "webnlg":
            con.replaced_items = test[1]
        elif con.corp == "e2e":
            con.input_kvs = test[1]
            con.possible_val = test[2]
        test = test[0]

    corp_list = list(train) + list(test)
    prefix = con.prefix

    return [deal_corp(con, prefix, x) for x in corp_list]
