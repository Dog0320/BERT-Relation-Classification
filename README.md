# BERT-Relation-Classification
BERT-Relation-Classification pytorch implementation

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

