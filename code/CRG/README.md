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

### `retrieve/retrieve.py`
Code to test various retrieval methods from question database. Implements the 4 classifiers, 3 feature extractors, and 5 retrieval methods, totaling to 32 run combinations (see note below)

Use: `python3 retrieve/retrieve.py` [args]
- `--classify_method [method]` classification method to use
  - `LR` linear regression
  - `SVM` support vector machine
  - `BERT` finetuned BERT
  - `DistilBERT` finetuned DistilBERT
- `--extract_method [method]` feature extraction method to use
  - `NER` - Name Entity Recognition
  - `TFIDF` - Term Frequency-Inverse Document Frequency
  - `vec` - Word Embeddings for Semantic Search
- `--retrieve_method [method]` answer retrieval method
  - `EKI` Extract Keyword Intersection
  - `Jaccard` Jaccard Similarity
  - `JEKI` weighted sum of EKI and Jaccard
  - `CSS-TFIDF` cosine similarity from vector representations by TF-IDF
  - `CSS-vec` cosine similarity from vector representations by Word2Vec
- `--run_study` if EKI or Jaccard, run 5 times and average scores

Note 1: There are two main retrieval methods: keyword comparison and vector space comparison. This means *there are invalid combinations* of extractions methods and retrieval methods. The valid combinations below:
- {NER, TFIDF} $\mapsto$ {EKI, Jaccard, JEKI}
- {TFIDF, vec} $\mapsto$ {CSS-TFIDF, CSS-vec}

Note 2: To use the CSS retrieval method, the associated extraction method must be used (i.e. `CSS-TFIDF` use `TFIDF` extract method, `CSS-vec` use `vec` extract method). This is due to where the vectorizers are initialized. That's a fix for later.

Note 3: The `run_study` is for EKI or Jaccard only since their output is non-deterministic. To reflect some type of typical behavior, it is run 5 times and averaged. 