import argparse
from utils.set_seed import set_seed
from utils.device import device


def config(train, args=None):
    parser = argparse.ArgumentParser("Config")

    parser.add_argument("--aspect",
                        choices=["systematicity", "productivity", "order-invariance",
                                 "substitutivity", "rule-learnability"], default="systematicity",
                        help="The aspect to be evaluated.")

    parser.add_argument("--sys",
                        choices=["atom", "combination"], default="atom",
                        help="The systematicity training set.")

    parser.add_argument("--sys_co_limit", type=float, default=0.02,
                        help="The systematicity co_limit.")

    parser.add_argument("--pro", choices=["invisible", "visible", "full"], default="invisible",
                        help="The productivity training set.")

    parser.add_argument("--pro_len", type=int, default=4,
                        help="The max seen length in productivity training set \"invisible\".")

    parser.add_argument("--pro_co_limit", type=float, default=0.02,
                        help="The productivity co_limit.")

    parser.add_argument("--ord", choices=["match", "original"], default="original",
                        help="The order-invariance training set.")

    parser.add_argument("--ord_aspect", choices=["main", "cwio", "performance"], default="main",
                        help="The aspect of order-invariance to be evaluated.")

    parser.add_argument("--check_dataset", action="store_true")

    parser.add_argument("--model",
                        # choices=["t5-large", "facebook/bart-large", "gpt2-large",
                        #          "flan-t5-xxl", "Llama-2-13b-hf", "Mistral-7B-v0.1"],
                        default="t5-large")

    parser.add_argument("--corp", choices=["webnlg", "e2e"], default="webnlg",
                        help="The corpus used.")

    parser.add_argument("--corp_folder", default="../webnlg3.0/",
                        help="The folder of the original corpus.")

    parser.add_argument("--dataset_folder", default="../dataset_webnlg/",
                        help="The folder of the constructed dataset.")

    parser.add_argument("--model_folder", default="",
                        help="The folder of the original model (Default: download from HuggingFace).")

    parser.add_argument("--trained_model_folder", default="../model/",
                        help="The folder for saving trained models.")

    parser.add_argument("--res_folder", default="../result/",
                        help="The folder for saving results.")

    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--in_size_test", default=18, type=int)
    parser.add_argument("--in_size_train", default=6, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_decay", default=False)
    parser.add_argument("--beam_width", default=5)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--norm", default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--patience", default=5)
    parser.add_argument("--stop_method", choices=["epoch", "patience"], default="epoch")
    parser.add_argument("--without_dev", default=False)
    parser.add_argument("--drop_rate", default=0.0)

    parser.add_argument("--prefix", default="translate from Triple to Text: ")

    parser.add_argument("--device", default="0")
    parser.add_argument("--pre_check_mode", action="store_true")
    parser.add_argument("--pre_check_len", default=100)

    if not args:
        con = parser.parse_args( )
    else:
        args = args.split(" ")
        con = parser.parse_args(args)

    if con.corp == "e2e":
        con.corp_folder = "../e2e-clean/"
        con.dataset_folder = "../dataset_e2e/"
        con.prefix = "translate from MR to Text: "

    con.write_test_result_in_training = (con.aspect != "rule-learnability")

    if train:
        if con.aspect == "order-invariance":
            con.ord_aspect = "performance"

    # Use smaller in_size_train / in_size_test to avoid OOM
    in_train_size_index = {
        "flan-t5-xxl": (6, 6),
        "Llama-2-13b-hf": (3, 3),
        "Mistral-7B-v0.1": (6, 6),
        "gpt2-large": (6, 8)
    }

    if con.model in in_train_size_index.keys( ):
        con.in_size_train = in_train_size_index[con.model][0]
        con.in_size_test = in_train_size_index[con.model][1]

    if con.corp == "webnlg" and con.model == "flan-t5-xxl":
        if con.aspect != "rule-learnability":
            con.in_size_train = 3
        con.in_size_test -= 1

    con.in_size_train = min(con.in_size_train, con.batch_size)
    con.div_factor = con.batch_size / con.in_size_train

    con.device = device("cuda:" + con.device)
    set_seed(con.seed)
    con.train = train

    con.score_type = "performance"

    if not con.train:

        if con.aspect == "order-invariance":
            if con.ord_aspect == "main":
                con.score_type = "order-main"
            if con.ord_aspect == "cwio":
                con.score_type = "order-cwio"

        if con.aspect == "rule-learnability":
            con.score_type = "rate"

    def check_type(x):
        if "t5" in x or "bart" in x:
            return "seq2seq"
        elif "gpt2" in x or "Llama" in x or "Mistral" in x:
            return "causal"
        else:
            return "causal"

    con.model_type = check_type(con.model)
    con.use_lora = True

    con.parent_lambda = 0.5

    con.extra_tokens = [ ]
    con.tokenizer = None
    con.data_name = None

    # For rule-learnability
    con.ent_unk = "Entity"
    con.val_unk = "Value"
    con.replaced_items = None
    con.input_kvs = None
    con.possible_val = None

    # For order-invariance
    con.test_plans = None
    con.test_pairs = None

    return con
