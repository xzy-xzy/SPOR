class Entry:
    def __init__(self, e, mr=False):
        self.data = e
        self.triples = [ ]
        if not mr:
            for x in e.modifiedtripleset.triples:
                self.triples.append((x.s, x.p, x.o))
        else:
            for x in e.modifiedtripleset.triples:
                self.triples.append(("None", x.p, x.o))
        self.texts = [ ]
        for x in e.lexs:
            self.texts.append(x.lex)