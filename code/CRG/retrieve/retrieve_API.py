'''
CRG - Retrieve Research

Author: Ian Jackson
Date: 03-09-2025

'''

#== Imports ==#
import os
import sys
import json
import time
import argparse
import torch
import spacy
import random

sys.path.insert(1, './classify')

import torch.nn as nn
import torch.optim as optim
import numpy as np
import traditional_ML as trad           #type: ignore
import finetune_transformer as trans    #type: ignore

from typing import Union
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader

#== Global Variables ==#
test_dataset = {'data': [
    {'question': 'What degree programs does the department offer?', 'label': 'Degree Programs', 'correct_ans_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'question': 'What dual degrees can I pursue?', 'label': 'Degree Programs', 'correct_ans_idx': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
    {'question': 'What are the various research areas in the Lane Department?', 'label': 'Research Opportunities', 'correct_ans_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'question': 'What research is done in the biometrics field?', 'label': 'Research Opportunities', 'correct_ans_idx': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
    {'question': 'What are the student orgs I can join as a LCSEE student?', 'label': 'Clubs and Organizations', 'correct_ans_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'question': 'What kind of activities do CyberWVU students do?', 'label': 'Clubs and Organizations', 'correct_ans_idx': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]},
    {'question': 'What can I do with a computer engineering degree?', 'label': 'Career Opportunities', 'correct_ans_idx': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
    {'question': 'What can I do with a computer science degree?', 'label': 'Career Opportunities', 'correct_ans_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'question': 'What type of internships do students get?', 'label': 'Internships', 'correct_ans_idx': [6, 7, 8, 9]},
    {'question': 'How can students get internships?', 'label': 'Internships', 'correct_ans_idx': [0, 1, 2, 3, 4, 5]},
    {'question': 'What type of scholarships are available for incoming students?', 'label': 'Financial Aid and Scholarships', 'correct_ans_idx': [0, 1, 2, 3, 4, 5]},
    {'question': 'How can freshmen get scholarships?', 'label': 'Financial Aid and Scholarships', 'correct_ans_idx': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
    {'question': 'How can I get into contact with the Lane Department?', 'label': 'Location and Contact', 'correct_ans_idx': [3, 4, 5]},
    {'question': 'Where is the Lane Department Located?', 'label': 'Location and Contact', 'correct_ans_idx': [0, 1, 2]},
]}

spacy_model = None
tfidf_vectorizer = None
w2v_model = None

#== Classes ==#


#== Methods ==#
def load_dataset(path: str) -> dict:
    '''
    load the custom dataset and label the questions for classification

    Args:
        path (str): path to custom dataset

    Returns:
        dict: dataset with labeled questions
    '''
    # check if path exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    # load the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    data = data['data']

    # iterate through each label
    labeled_data = {'data': []}

    for label_data in data:
        # extract the label and qa data
        label = label_data['title']
        qas = label_data['paragraphs'][0]['qas']

        # extract each question, adding it to the labeled dataset
        for qa in qas:
            question = qa['question']

            labeled_data['data'].append(
                {
                    'question': question,
                    'label': label
                }
            )

    return labeled_data

def classify_and_extract(args: dict, question: str) -> Union[str, list]:
    '''
    use classification method and extraction method to fill the test dataset with
    their predicted class and keywords

    Args:
        args (dict): script arguments (classify_method and extract_method)
    '''
    # load the dataset
    dataset = load_dataset('../dataset.json')

    # output
    predicted_category = None
    kw = None

    #-- determine the classification used --#
    if args['classify_method']:
        #- Linear Regression -#
        if args['classify_method'] == 'LR':
            print('Using Linear Regression')

            # extract the questions and labels
            questions = [item['question'] for item in dataset['data']]
            labels = [item['label'] for item in dataset['data']]

            # convert text into TF-IDF features
            vectorizer = TfidfVectorizer(max_features=1000)
            X_tfidf = vectorizer.fit_transform(questions).toarray()

            # encode labels to numerical values
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(labels)

            # convert data into PyTorch Tensors
            X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)

            # initialize the model
            input_dim = X_tensor.shape[1]
            num_classes = len(set(labels))
            model = trad.LogisticRegression(input_dim, num_classes)

            print("Loading trained model")
            model.load_state_dict(torch.load("classify/lr_c_model.pth"))
            model.eval()

            # get predicted class 
            predicted_category = trad.classify_question(question, vectorizer, model, label_encoder, False)

        #- Support Vector Machine -#
        elif args['classify_method'] == 'SVM':
            print('Using Support Vector Machine')

            # extract the questions and labels
            questions = [item['question'] for item in dataset['data']]
            labels = [item['label'] for item in dataset['data']]

            # convert text into TF-IDF features
            vectorizer = TfidfVectorizer(max_features=1000)
            X_tfidf = vectorizer.fit_transform(questions).toarray()

            # encode labels to numerical values
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(labels)

            # convert data into PyTorch Tensors
            X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)

            # initialize the model
            input_dim = X_tensor.shape[1]
            num_classes = len(set(labels))
            model = trad.SVM(input_dim, num_classes)

            print("Loading trained model")
            model.load_state_dict(torch.load("classify/svm_c_model.pth"))
            model.eval()

            # get prediction
            predicted_category = trad.classify_question(question, vectorizer, model, label_encoder, False)

        #- BERT -#
        elif args['classify_method'] == 'BERT':
            print('Using BERT')

            # preprocess the dataset
            train_labels = [item['label'] for item in dataset['data']]

            # encode labels to numerical values
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(train_labels)
            
            model = BertForSequenceClassification.from_pretrained(f"./classify/bert-question-classifier")
            tokenizer = BertTokenizer.from_pretrained(f"./classify/bert-question-classifier")

            # evaluate BERT on test dataset
            model.eval()

            # tokenize question
            inputs = tokenizer(question, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()

            # map to label
            predicted_category = label_encoder.inverse_transform([prediction])[0]

        #- DistilBERT -#
        elif args['classify_method'] == 'DistilBERT':
            print('Using DistilBERT')

            # preprocess the dataset
            train_labels = [item['label'] for item in dataset['data']]

            # encode labels to numerical values
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(train_labels)

            model = DistilBertForSequenceClassification.from_pretrained(f"./classify/distilbert-question-classifier")
            tokenizer = DistilBertTokenizer.from_pretrained(f"./classify/distilbert-question-classifier")

            # evaluate DistilBERT on test dataset
            model.eval()


            # tokenize question
            inputs = tokenizer(question, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()

            # map to label
            predicted_category = label_encoder.inverse_transform([prediction])[0]

        else:
            print('Error: Classification method not defined')
            quit()
    else: 
        print('Error: Provide classification method')
        quit()

    #-- determine extraction method --#
    # put keywords associated with question into test dataset 
    if args['extract_method']:
        #- NER -#
        if args['extract_method'] == 'NER':
            print('Using NER')

            # load pre-trained NER model
            nlp = spacy.load("en_core_web_sm")
            global spacy_model
            spacy_model = nlp

            # extract keywords for each test question 
            kw = trad.extract_keywords_NER(question, nlp)

        #- TF-IDF -#
        elif args['extract_method'] == 'TFIDF':
            print('Using TF-IDF')

            # convert questions to TF-IDF features
            vectorizer = TfidfVectorizer(stop_words='english')
            X_tfidf = vectorizer.fit_transform([item['question'] for item in test_dataset['data']])

            global tfidf_vectorizer
            tfidf_vectorizer = vectorizer

            # extract keywords for each test question 
            kw = trad.extract_keywords_TFIDF(question, vectorizer)

        #- Word2Vec -#
        elif args['extract_method'] == 'vec':
            print('Using Word2Vec')

            # load a pretrained model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            global w2v_model
            w2v_model = model

            # extract keywords for each test question 
            kw = model.encode(question)
    else: 
        print('Error: Provide extraction method')
        quit()

    return predicted_category, kw

def filter_dataset(classification: str) -> dict:
    '''
    given a class, return the QA pairs associated with only that class

    Args:
        classification (str): question class

    Returns:
        dict: QA pairs under given class
    '''
    # check if path exist
    if not os.path.exists('../dataset.json'):
        raise FileNotFoundError(f"Dataset not found at {'../dataset.json'}")
    
    # load the JSON file
    with open('../dataset.json', 'r') as f:
        data = json.load(f)

    data = data['data']

    for chunk in data:
        if chunk['title'] == classification:
            return chunk['paragraphs'][0]['qas']

    print(f'Error: class ({classification}) not found in dataset')
    quit()

def retrieve_EKI(filtered_data: dict, keywords: list) -> str:
    '''
    retrieve answer using exact keyword intersection

    Args:
        filtered_data (dict): filtered QA pairs based on class
        keywords (list): keywords extracted from question

    Returns:
        str: predicted answer
    '''
    # tag each question with its score
    for i,qa in enumerate(filtered_data):
        question = qa['question']
        score = 0

        # for each keyword, see if its in the question
        for kw in keywords:
            # keywords can be either strings or tuples
            if type(kw) is str:
                if kw in question: score += 1
            elif type(kw) is tuple:
                x,y = kw
                if x in question: score += 1
                if y in question: score += 1

        filtered_data[i]['score'] = score

    # get the highest scoring answers
    max_score = max(item['score'] for item in filtered_data)
    best_ans = [item['answer'] for item in filtered_data if item['score'] == max_score]
    
    # flatten 
    best_ans = [item[0] for item in best_ans]

    # if one answer, use that; otherwise choose randomly
    if len(best_ans) == 1:
        return best_ans[0]
    else:
        return best_ans[random.randint(0, len(best_ans) - 1)]

def retrieve_Jaccard(filtered_data: dict, keywords: list, kw_method: str = 'NER') -> str:
    '''
    retrieve answer using Jaccard similarity (IoU) 

    Args:
        filtered_data (dict): filtered QA pairs based on class
        keywords (list): keywords extracted from question
        kw_method (str): keyword extraction method (default NER)

    Returns:
        str: predicted answer
    '''
    for i,qa in enumerate(filtered_data):
        question = qa['question']
        question_kw = None

        # get keywords based on extraction method
        if kw_method == 'NER':
            question_kw = trad.extract_keywords_NER(question, spacy_model)
        elif kw_method == 'TFIDF':
            question_kw = trad.extract_keywords_TFIDF(question, tfidf_vectorizer)
        else:
            print('Error: invalid extraction method in Jaccard similarity')
            quit()

        # flatten with possibility of tuples
        keywords = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in keywords))
        question_kw = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_kw))

        # compute Jaccard similarity score
        intersection = set(keywords) & set(question_kw)
        union = set(keywords) | set(question_kw)
        score = len(intersection) / len(union) if union else 0

        # store score in dataset
        filtered_data[i]['score'] = score

    # get the highest scoring answers
    max_score = max(item['score'] for item in filtered_data)
    best_ans = [item['answer'] for item in filtered_data if item['score'] == max_score]
    
    # flatten 
    best_ans = [item[0] for item in best_ans]

    # if one answer, use that; otherwise choose randomly
    if len(best_ans) == 1:
        return best_ans[0]
    else:
        return best_ans[random.randint(0, len(best_ans) - 1)]

def retrieve_JEKI(filtered_data: dict, keywords: list, kw_method: str = 'NER', lamb_1: float = 0.5, lambd_2: float = 0.5) -> str:
    '''
    retrieve answer using JEKI (weighted sum of EKI and Jaccard)

    Args:
        filtered_data (dict): filtered QA pairs based on class
        keywords (list): keywords extracted from question
        kw_method (str): keyword extraction method (default NER)
        lambd_1 (float): weight for EKI score (default 0.5)
        lambd_2 (float): weight for Jaccard score (default 0.5)

    Returns:
        str: predicted answer
    '''
    for i,qa in enumerate(filtered_data):
        question = qa['question']
        question_kw = None

        # get keywords based on extraction method
        if kw_method == 'NER':
            question_kw = trad.extract_keywords_NER(question, spacy_model)
        elif kw_method == 'TFIDF':
            question_kw = trad.extract_keywords_TFIDF(question, tfidf_vectorizer)
        else:
            print('Error: invalid extraction method in Jaccard similarity')
            quit()

        # flatten with possibility of tuples
        keywords = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in keywords))
        question_kw = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_kw))

        # compute Jaccard similarity score
        intersection = set(keywords) & set(question_kw)
        union = set(keywords) | set(question_kw)
        jaccard_score = len(intersection) / len(union) if union else 0

        # compute EKI score
        EKI_score = 0

        # for each keyword, see if its in the question
        for kw in keywords:
            # keywords can be either strings or tuples
            if type(kw) is str:
                if kw in question: EKI_score += 1
            elif type(kw) is tuple:
                x,y = kw
                if x in question: EKI_score += 1
                if y in question: EKI_score += 1

        # store score in dataset
        filtered_data[i]['score'] = lamb_1 * EKI_score + lambd_2 * jaccard_score

    # get the highest scoring answers
    max_score = max(item['score'] for item in filtered_data)
    best_ans = [item['answer'] for item in filtered_data if item['score'] == max_score]
    
    # flatten 
    best_ans = [item[0] for item in best_ans]

    # if one answer, use that; otherwise choose randomly
    if len(best_ans) == 1:
        return best_ans[0]
    else:
        return best_ans[random.randint(0, len(best_ans) - 1)]
    
def retrieve_CSC_TFIDF(filtered_data: dict, ask_question: str) -> str:
    '''
    retrieve answer using cosine similarity with TD-IDF embeddings 

    Args:
        filtered_data (dict): filtered QA pairs based on class
        ask_question (str): test question/sample question

    Returns:
        str: predicted answer
    '''
    # for each question in filtered data, get their vector embedding
    for i,qa in enumerate(filtered_data):
        db_question = qa['question']
        tdidf_matrix = tfidf_vectorizer.transform([ask_question, db_question])
        cosine_sim_matrix = cosine_similarity(tdidf_matrix, tdidf_matrix)

        # extract score
        css = cosine_sim_matrix[0,1]

        # store score in dataset
        filtered_data[i]['score'] = css

    # get the best scoring answer
    # get the highest scoring answers
    max_score = max(item['score'] for item in filtered_data)
    best_ans = [item['answer'] for item in filtered_data if item['score'] == max_score]
    
    # flatten 
    best_ans = [item[0] for item in best_ans]

    # if one answer, use that; otherwise choose randomly
    if len(best_ans) == 1:
        return best_ans[0]
    else:
        return best_ans[random.randint(0, len(best_ans) - 1)]

def retrieve_CSC_vec(filtered_data: dict, ask_question: str) -> str:
    '''
    retrieve answer using cosine similarity with TD-IDF embeddings 

    Args:
        filtered_data (dict): filtered QA pairs based on class
        ask_question (str): test question/sample question

    Returns:
        str: predicted answer
    '''
    # get vec embedding of asked question
    ask_question_vec = w2v_model.encode(ask_question)
    ask_question_vec = np.array(ask_question_vec).reshape(1, -1)

    # for each question in filtered data, get their vector embedding
    for i,qa in enumerate(filtered_data):
        db_question = qa['question']
        db_question_vec = w2v_model.encode(db_question)
        db_question_vec = np.array(db_question_vec).reshape(1, -1)

        # compute similarity
        css = cosine_similarity(ask_question_vec, db_question_vec)[0, 0]

        # store score in dataset
        filtered_data[i]['score'] = css

    # get the best scoring answer
    # get the highest scoring answers
    max_score = max(item['score'] for item in filtered_data)
    best_ans = [item['answer'] for item in filtered_data if item['score'] == max_score]
    
    # flatten 
    best_ans = [item[0] for item in best_ans]

    # if one answer, use that; otherwise choose randomly
    if len(best_ans) == 1:
        return best_ans[0]
    else:
        return best_ans[random.randint(0, len(best_ans) - 1)]

#== Main Execution ==#
def main(args: dict, question: str):
    # fill test dataset with predictions using provided classification method and extraction method
    pred_class, kw = classify_and_extract(args, question)

    #-- extract the answer from the database --#

    # get filtered data from class
    filtered_data = filter_dataset(pred_class)
    
    # get best answer (based on method)
    #- EKI -#
    if args['retrieve_method'] == 'EKI':
        # retrieve answer
        answer = retrieve_EKI(filtered_data, kw)

    #- Jaccard -#
    elif args['retrieve_method'] == 'Jaccard':
        # retrieve answer
        extract_method = 'TFIDF' if args['extract_method'] == 'TFIDF' else 'NER'

        answer = retrieve_Jaccard(filtered_data, kw, extract_method)

    #- JEKI -#
    elif args['retrieve_method'] == 'JEKI':
        # retrieve answer
        extract_method = 'TFIDF' if args['extract_method'] == 'TFIDF' else 'NER'

        answer = retrieve_JEKI(filtered_data, kw, extract_method)

    #- Cosine Similarity using TFIDF -#
    elif args['retrieve_method'] == 'CSS-TFIDF':
        # retrieve answer
        answer = retrieve_CSC_TFIDF(filtered_data, question)

    #- Cosine Similarity using WOrd2Vec -#
    elif args['retrieve_method'] == 'CSS-vec':
        # retrieve answer
        answer = retrieve_CSC_vec(filtered_data, question)

    print("=== SUMMARY ===")
    print(f"Classification = {args['classify_method']} | Extraction = {args['extract_method']} | Retrieve = {args['retrieve_method']}")

    return answer

if __name__ == "__main__":
    args = {}

    main(args)