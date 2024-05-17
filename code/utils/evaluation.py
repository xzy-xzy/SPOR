import os
from utils.data_match import get_kendall, get_oc
from parent import parent
from parent.check import paraphase_check
from utils.nltk_wordtokenizer import tokenizer
from string import punctuation


def evaluation(pack, con, output_folder, rec_name):
    # performance: calculate PARENT
    if con.score_type == "performance":
        f = open(output_folder + rec_name + ".txt", "w")
        for x in pack["preds"]:
            f.write(x.strip( ) + "\n")
        preds, refs, data = pack["preds"], pack["refs"], pack["data"]
        preds = [tokenizer(x.lower( )) for x in preds]
        refs = [[tokenizer(x.lower( )) for x in y] for y in refs]

        if con.corp == "webnlg":
            data = [[tuple([tokenizer(x.lower( )) for x in y]) for y in z if paraphase_check(y)] for z in data]
        elif con.corp == "e2e":
            def convert_to_attr_value(y):
                ret = [tokenizer(x.lower( )) for x in y]
                assert len(ret) == 3
                return tuple([ret[1], ret[2]])
            data = [[convert_to_attr_value(y) for y in z] for z in data]

        precision, recall, f_score = parent(preds, refs, data, avg_results=True,
                                            n_jobs=1, lambda_weight=con.parent_lambda)
        info = f"PARENT = %.4f\n" % f_score
        print(info)
        return [info, f_score]

    # order-main: [Order invariance] calculate PBH / PBH on fidelity (ext) and ordering (ord)
    if con.score_type == "order-main":
        f = open(output_folder + rec_name + "_1.txt", "w")
        for x in pack["output1"]:
            f.write(x.strip( ) + "\n")
        g = open(output_folder + rec_name + "_2.txt", "w")
        for x in pack["output2"]:
            g.write(x.strip( ) + "\n")
        cnt = 0
        only_one_ext, only_one_ord = 0, [0, 0]
        both_ext, both_ord = 0, [0, 0]
        strict = (con.corp == "e2e")    # For limited values, we use strict matching
        for o1, o2, p, pr in zip(pack["output1"], pack["output2"], con.test_plans, con.test_pairs):
            cnt += 1
            ext1, ord1, pl1 = get_oc(o1, p, pr, strict)
            ext2, ord2, pl2 = get_oc(o2, p, pr, strict)
            only_one_ext += (ext1 != ext2)
            both_ext += (ext1 and ext2)
            for i in range(2):
                only_one_ord[i] += (ord1[i] != ord2[i])
                both_ord[i] += (ord1[i] and ord2[i])
        info = "Both_Ext = %.4f\nOnly_One_Ext = %.4f\n" \
               % (both_ext / cnt * 100, only_one_ext / cnt * 100) + \
               "Both_Ord(EM) = %.4f\nOnly_One_Ord(EM) = %.4f\n" \
               % (both_ord[0] / cnt * 100, only_one_ord[0] / cnt * 100) + \
               "Both_Ord(KT) = %.4f\nOnly_One_Ord(KT) = %.4f\n" \
               % (both_ord[1] / cnt * 100, only_one_ord[1] / cnt * 100)
        print(info)
        return [info, None]

    # order-cwio: [Order invariance] calculate CWIO
    elif con.score_type == "order-cwio":
        f = open(output_folder + rec_name + ".txt", "w")
        for x in pack["output"]:
            f.write(x.strip( ) + "\n")
        d, tot_n, tot_occ = [ ], 0, 0
        for i in range(7):
            d.append([ ])
        for o, p in zip(pack["output"], con.test_pairs):
            n, cor, occ = get_kendall(o, p)
            if n <= 7:
                tot_n += n
                tot_occ += occ
                d[n - 1].append(cor)
        info, macro = "", [ ]
        for i in range(7):
            if not d[i]:
                continue
            dis = sum(d[i]) / len(d[i])
            info += "KD-%d = %.4f\n" % (i + 1, dis)
            macro.append(dis)
        macro = sum(macro) / len(macro)
        info += "KD-Macro = %.4f\n" % macro
        info += "Occ_Rate = %.4f\n" % (tot_occ / tot_n * 100)
        print(info)
        return [info, None]

    # rate: [Rule learnability] calculate proportions of four cases
    elif con.score_type == "rate":
        f = open(output_folder + rec_name + ".txt", "w")
        for x in pack["output"]:
            f.write(x.strip( ) + "\n")
        g = open(output_folder + "error.txt", "w")
        inc_10, inc_01, inc_11, inc_00, inc_all = 0, 0, 0, 0, 0

        def remove_extra_blank(s):
            t = [ ]
            for i in range(len(s)):
                if s[i] == ' ' and (i == 0 or s[i - 1] == ' '):
                    continue
                t.append(s[i])
            return ''.join(t)

        def remove_punc(s):
            return "".join([x for x in s if x not in punctuation])

        if con.corp == "webnlg":
            conv = {"entity 1": "1st entity", "entity 2": "2nd entity", "entity 3": "3rd entity",
                    "entity 4": "4th entity", "entity 5": "5th entity", "entity 6": "6th entity",
                    "entity 7": "7th entity", "entity 8": "8th entity", "entity 9": "9th entity",
                    "entity 10": "10th entity"}
            for o, kv, d in zip(pack["output"], con.replaced_items, pack["data"]):
                o, u, e = o.lower( ), True, False
                o = remove_extra_blank(o)
                piece = set(remove_punc(o).split( ))    # For fuzzy matching: Entity 1 -> 1
                for k, v in kv.items( ):
                    if o.find(k.lower( )) != -1:
                        e = True
                    if o.find(v.lower( )) == -1 and o.find(conv[v.lower( )]) == -1 and v[-1] not in piece:
                        u = False
                inc_all += 1
                inc_10 += u and (not e)
                inc_01 += e and (not u)
                inc_11 += u and e
                inc_00 += (not u) and (not e)
                if e and (not u):
                    g.write(f"-EU* {o}\n--Input: {d}\n--Entity: {kv}\n")
                if e and u:
                    g.write(f"-E* {o}\n--Input: {d}\n--Entity: {kv}\n")
                if (not e) and (not u):
                    g.write(f"-U* {o}\n--Input: {d}\n--Entity: {kv}\n")

        elif con.corp == "e2e":
            for o, kv in zip(pack["output"], con.input_kvs):
                o, u, e = o.lower( ), True, False
                o = remove_extra_blank(o)
                for k, v in kv:
                    v_f = v.replace(con.val_unk + " ", "")    # For fuzzy matching: Value A -> A
                    if o.find(v_f.lower( )) == -1 and o.find(v.lower( )) == -1:
                        u = False
                set_k = set([x[0] for x in kv])
                e_k, e_c = None, None
                for k in con.possible_val:
                    if k in set_k:
                        for candi in con.possible_val[k]:
                            if o.find(candi.lower( )) != -1:
                                e_k, e_c, e = k, candi, True
                inc_all += 1
                inc_10 += u and (not e)
                inc_01 += e and (not u)
                inc_11 += u and e
                inc_00 += (not u) and (not e)
                if e and (not u):
                    g.write(f"-EU*\n{o}\n{kv}\n{e_k} {e_c}\n")
                if e and u:
                    g.write(f"-E*\n{o}\n{kv}\n{e_k} {e_c}\n")
                if (not e) and (not u):
                    g.write(f"-U*\n{o}\n{kv}\n")

        rate_10 = inc_10 / inc_all * 100
        rate_01 = inc_01 / inc_all * 100
        rate_11 = inc_11 / inc_all * 100
        rate_00 = inc_00 / inc_all * 100
        info = "(1,0)_Rate = %.4f\n(0,1)_Rate = %.4f\n(1,1)_Rate = %.4f\n(0,0)_Rate = %.4f\n" \
               % (rate_10, rate_01, rate_11, rate_00)
        print(info)
        return [info, rate_10]

