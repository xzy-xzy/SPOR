# SPOR
Dataset and code for paper [SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation](https://arxiv.org/abs/2405.10650).

## Dataset
The datasets are available as `dataset.jsonl` files in `dataset_webnlg` and `dataset_e2e`. The fields included in the file are:
- `data`: `list[list[str]]`. Input data units.
- `texts`: `list[str]`. Reference texts.
- `input_order_1 / input_order_2`: `list[int]`. The input order of the data units. Data units are indicated by their index in `data`. The fields appear only in `order-invariance_main`.
- `output_order`: `list[list[int]]`. The order of occurrence of data units in the reference texts. The field appear only in `order-invariance_main`.
- `replaced_entity`: `dict[str, str]`. Replaced entities and their representation after replacement. The field appear only in `test_webnlg_rule-learnability`.
- `replaced_kv_pair`: `dict[str, str]`. The attributes to which the replaced values correspond, and the representations of the replaced values. The field appear only in `test_e2e_rule-learnability`, with `possible_value.json` to indicate all possible values corresponding to the attribute.


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
Welcome to cite our work if it helps.
```
@misc{xu2024sporcomprehensivepracticalevaluation,
      title={SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation}, 
      author={Ziyao Xu and Houfeng Wang},
      year={2024},
      eprint={2405.10650},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.10650}, 
}
```


## Acknowledgements

* The code of PARENT metric is based on [KaijuML/parent](https://github.com/KaijuML/parent).
* The code of corpus-reader is based on [WebNLG/corpus-reader](https://gitlab.com/webnlg/corpus-reader).
