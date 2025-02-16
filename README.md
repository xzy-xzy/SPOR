# SPOR
Dataset and code for the paper [SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation](https://arxiv.org/abs/2405.10650).

## Dataset
The datasets are available as `dataset.jsonl` files in `dataset_webnlg` and `dataset_e2e`. The key-value pairs included in the file are:
- `data`: `list[list[str]]`. Input data units.
- `texts`: `list[str]`. Reference texts.
- `input_order_1 / input_order_2`: `list[int]`. The input order of the data units. Data units are indicated by their index in `data`. The pairs appear only in `order-invariance_main`.
- `output_order`: `list[list[int]]`. The order of occurrence of data units in the reference texts. The pairs appear only in `order-invariance_main`.
- `replaced_entity`: `dict[str, str]`. Replaced entities and their representation after replacement. The pairs appear only in `test_webnlg_rule-learnability`.
- `replaced_kv_pair`: `dict[str, str]`. The attributes to which the replaced values correspond, and the representations of the replaced values. The pairs appear only in `test_e2e_rule-learnability`, with `possible_value.json` to indicate all possible values corresponding to the attribute.


## Code
### Getting Started
Start by:
```
cd code
```
Install required packages with: 
```
pip install -r requirements.txt
```
### Dataset construction
- See `code/prepare_corpus.sh` for commands for dataset construction.
- See `code/construction` for the specific code for dataset construction.
- If you want to reproduce the construction process, first delete the existing dataset files. 

### Training and Evaluation
See `code/run.sh` for commands for training and evaluation.

## Citation
```
@inproceedings{xu-wang-2024-spor,
    title = "{SPOR}: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation",
    author = "Xu, Ziyao  and
      Wang, Houfeng",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.36",
    doi = "10.18653/v1/2024.acl-long.36",
    pages = "604--621",
}
```


## Acknowledgements

* The code of PARENT metric is based on [KaijuML/parent](https://github.com/KaijuML/parent).
* The code of corpus-reader is based on [WebNLG/corpus-reader](https://gitlab.com/webnlg/corpus-reader).
