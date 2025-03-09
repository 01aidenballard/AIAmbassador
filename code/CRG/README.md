# Classify-Retrieve-Generate (CRG)
## Setup
Create a python virtual environment named `venv`
``` 
python3.10 -m venv venv
```
Activate by `source venv/bin/activate`

Install the required libraries 
```
pip3 install ...
```

## Python Files
### `classify/traditional_ML.py`
Code to train and run LR or SVM models, as well as extracting key information from the question.

Use: `python3 traditional_ML.py [args]`
- `--LR` run LR on the test dataset (if no saved model found, will train)
- `--SVM` run SVM on the test dataset (if no saved model found, will train)
- `--force` force training for selected ML model (must have `--LR` or `--SVM` flag)
- `--extract [method]` run feature/information extraction using passed method. Methods:
  - `NER` - Name Entity Recognition
  - `TFIDF` - Term Frequency-Inverse Document Frequency
  - `vec` - Word Embeddings for Semantic Search

### `classify/finetune_transformer.py`
Code to train and run BERT or DistilBERT. Note: finetuning these transformers takes a lot of processing power. This training was on using Google Colabs.

Use: `python3 finetune_transformer.py [args]`
- `--BERT` run BERT on test dataset (if not model found, will train)
- `--DistilBERT` run DistilBERT on test dataset (if not model found, will train)
- `--force` force training for selected ML model (must have `--BERT` or `--DistilBERT` flag)