corp="webnlg" # webnlg or e2e
model="t5-large"  # model evaluated

# The model name defaults to the repository name on HuggingFace.
# If you want to use a local model, specify the root directory of the local model with --model_folder
# and specify the folder of the local model with --model.
# E.g., to use model in "../../cache/t5-large", specify "--model_folder ../../cache/ ---model t5-large".

# systematicity
sys="atom"  # atom or combination
python3 train.py --aspect systematicity --sys $sys --corp $corp --model $model
python3 inference.py --aspect systematicity --sys $sys --corp $corp --model $model

# productivity
pro="invisible"  # invisible or visible
pro_len=3   # 3, 4, or 5
python3 train.py --aspect productivity --pro $pro --pro_len $pro_len --corp $corp --model $model
python3 inference.py --aspect productivity --pro $pro --pro_len $pro_len --corp $corp --model $model

# order-invariance
ord="original"  # original or match
ord_aspect="main"  # main, cwio, or performance
python3 train.py --aspect order-invariance --ord $ord --corp $corp --model $model
python3 inference.py --aspect order-invariance --ord $ord --ord_aspect $ord_aspect --corp $corp --model $model

# rule-learnability
python3 train.py --aspect rule-learnability --corp $corp --model $model
python3 inference.py --aspect rule-learnability --corp $corp --model $model


