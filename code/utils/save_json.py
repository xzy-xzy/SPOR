from os.path import exists
from os import makedirs
import json


def base_sample(e, corp, with_text=True):
    sample = { }
    triples = [[x.s, x.p, x.o] for x in e.modifiedtripleset.triples]
    if corp == "webnlg":
        sample["data"] = triples
    elif corp == "e2e":
        sample["name"] = triples[0][0]
        sample["data"] = [[x[1], x[2]] for x in triples]
    if with_text:
        sample["texts"] = [t.lex for t in e.lexs]
    return sample


def match_order(e, ori_e):
    e = [(x.s, x.p, x.o) for x in e.modifiedtripleset.triples]
    ori_e = [(x.s, x.p, x.o) for x in ori_e.modifiedtripleset.triples]
    return [ori_e.index(x) for x in e]


def save(f, sample):
    f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_train_json(root, corp, con):
    root = root + "dataset.jsonl"
    f = open(root, "w")
    for e in corp:
        sample = base_sample(e, con.corp)
        save(f, sample)


def save_test_json(root, corp, con):
    o_root = root
    root = root + "dataset.jsonl"
    f = open(root, "w")
    if con.aspect in ["systematicity", "productivity"]:
        for e in corp[0]:
            sample = base_sample(e, con.corp)
            save(f, sample)
    elif con.aspect == "order-invariance":
        if con.ord_aspect == "main":
            for idx, (e1, e2, order, ori_e) in enumerate(zip(corp[0][0], corp[0][1], corp[1], corp[3])):
                sample = base_sample(ori_e, con.corp)
                sample["input_order_1"] = match_order(e1, ori_e)
                sample["input_order_2"] = match_order(e2, ori_e)
                sample["output_order"] = order
                save(f, sample)
        elif con.ord_aspect == "cwio":
            for idx, e in enumerate(corp[0][0]):
                sample = base_sample(e, con.corp)
                save(f, sample)
        elif con.ord_aspect == "performance":
            for idx, e in enumerate(corp[0]):
                sample = base_sample(e, con.corp)
                save(f, sample)
    elif con.aspect == "rule-learnability":
        if con.corp == "webnlg":
            for e, replaced in zip(corp[0][0], corp[1]):
                sample = base_sample(e, con.corp, with_text=False)
                sample["replaced_entity"] = replaced
                save(f, sample)
        elif con.corp == "e2e":
            for e, kv in zip(corp[0][0], corp[1]):
                sample = base_sample(e, con.corp, with_text=False)
                sample["replaced_kv_pair"] = {x[0]: x[1] for x in kv}
                save(f, sample)
            root = o_root + "possible_value.json"
            f = open(root, "w")
            possible_value = {x: [z for z in y] for x, y in corp[2].items( )}
            f.write(json.dumps(possible_value, ensure_ascii=False))






