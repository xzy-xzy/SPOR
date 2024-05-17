from config import config
from model_control import ModelControl
from model_execute import ModelExecute
from utils.dataset import create_loader
from utils.corpus import get_corpus
from utils.gen_name import gen_name


def preprocess(train, args=None):
    con = config(train, args)
    main_name, sub_name = gen_name(con)
    ctrl = ModelControl(main_name, con)
    corp = get_corpus(con)
    task = ModelExecute(con, ctrl)
    con.tokenizer = task.model.tz
    loader = create_loader(corp, con)
    ctrl.insert_task(task, sub_name)
    ctrl.load( )
    return con, ctrl, loader
