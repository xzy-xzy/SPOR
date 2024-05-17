from config import config
from utils.corpus import get_corpus
from utils.gen_name import gen_name

con = config(False)
gen_name(con)
get_corpus(con)
