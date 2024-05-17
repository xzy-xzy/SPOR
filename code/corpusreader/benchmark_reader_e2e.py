from utils.convert import decode_camel, bool_convert
from utils.data_match import get_oc
import csv


class Triple:

    def __init__(self, s, p, o):
        self.s = s
        self.o = o
        self.p = p

    def flat_triple(self):
        return self.s + ' | ' + self.p + ' | ' + self.o


class Tripleset:

    def __init__(self):
        self.triples = []
        self.clusterid = 0

    def fill_tripleset(self, mr):
        reps = mr.split(',')
        reps = [x.strip( ) for x in reps]
        kv = [ ]
        for x in reps:
            bracket = x.find('[')
            kv.append((x[:bracket], x[bracket + 1:-1]))
        s = kv[0][1]
        for p, o in kv[1:]:
            # print(s, p, o)
            p = decode_camel(p)
            if p == "name":
                continue
            triple = Triple(s, p, o)
            self.triples.append(triple)

    def fill_tripleset_from_kv(self, kv):
        for k, v in kv:
            if k == "name":
                s = v
        for k, v in kv:
            if k == "name":
                continue
            triple = Triple(s, k, v)
            self.triples.append(triple)


class Lexicalisation:

    def __init__(self, lex):
        self.lex = lex

    def chars_length(self):
        return len(self.lex)


class Entry:

    def __init__(self):
        self.category = "e2e"
        self.modifiedtripleset = Tripleset()
        self.lexs = []

    def fill_modifiedtriple(self, mr):
        self.modifiedtripleset.fill_tripleset(mr)

    def create_lex(self, lex, strict):
        lex = lex.replace("family-friendly", "family friendly")
        pairs = [bool_convert(x.p, x.o) for x in self.modifiedtripleset.triples]
        if strict and not get_oc(lex, None, pairs, True):
            return
        lex = Lexicalisation(lex)
        self.lexs.append(lex)


def get_benchmark_e2e(file, erase_unstrict=False):
    f = open(file, "r")
    reader = csv.reader(f)
    cluster = { }
    for row in reader:
        mr, ref = row[0], row[1]
        if mr == "mr":
            continue
        if "name" not in mr:
            continue
        if mr not in cluster:
            cluster[mr] = [ ]
        cluster[mr].append(ref)
    benchmark = [ ]
    possible_val = { }
    for x in cluster:
        e = Entry( )
        e.fill_modifiedtriple(x)
        if len(e.modifiedtripleset.triples) == 0:
            continue
        if len(e.modifiedtripleset.triples) > 7:
            continue    # at least one attribute has two different values
        for lex in cluster[x]:
            e.create_lex(lex, erase_unstrict)
        if len(e.lexs) == 0:
            continue
        benchmark.append(e)
        for x in e.modifiedtripleset.triples:
            if x.p not in possible_val:
                possible_val[x.p] = set( )
            possible_val[x.p].add(x.o)
    return benchmark
