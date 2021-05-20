# BERT-Relation-Classification
Pytorch implmentation for [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284).

## Requirements
```
Python3.x
transformers==3.1.0
torch==1.6.1+
argparse==1.4.0

```

## Prepare

* Download ``google_model.bin`` from [here](https://drive.google.com/drive/folders/11i463eaaVvBrulLzSmUHdjFRgO_txBnU?usp=sharing), and save it to the ``assets/`` directory.
* Download ``dataset`` from [here](https://drive.google.com/drive/folders/1X2VcAbJ89Oj7VTsTMh7jRuuXh1zcjTpO?usp=sharing), and save it to the ``data/`` directory.

### Model Training

Run example on SemEval-2020 Task 8 dataset.
```
python3 train.py --task_name SemEval
```
#### To use your own dataset,  modify the DataProcessor in ``data_utils.py``.

### Model Evaluation

Run example on SemEval-2020 Task 8 dataset.
```
python3 evaluate.py --task_name SemEval
```

### Model Prediction

Run example on ATIS dataset.
```
python3 predict.py --task_name SemEval
```

### Results


|Dataset        |Accuracy |macro F1 score |
|-------------|------------|------------|
|SemEval |   90.67  | 81.80     |
|Snips|     97.69 |  78.79 |

The result on SemEval is evaluated with bidirectional relation (18 relations) macro F1 score.

While the author use SemEval-2010 Task 8 official scorer script which eventually calculate unidirectional relation (9 relations) macro F1 score.
