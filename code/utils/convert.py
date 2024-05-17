punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，–]+'


def convert(s):
    s = s.replace('_', ' ')
    s = s.replace('\\', '')
    s = s.replace('\"', '')
    return s


def so_conv(s):
    s = convert(s)
    t = [ ]
    for x in s:
        if x in punc:
            t.append(' ')
        else:
            t.append(x)
    return ''.join(t).strip( ).lower( )


def decode_camel(s):
    piece, tmp = [ ], [ ]
    for x in s:
        if x.isupper( ):
            piece.append("".join(tmp))
            tmp = [x.lower( )]
        else:
            tmp.append(x)
    if tmp:
        piece.append("".join(tmp))
    return " ".join(piece)


def bool_convert(p, o):
    if o not in ["no", "yes"]:
        return "none", o
    else:
        return p, "none"

