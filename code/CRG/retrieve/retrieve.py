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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader

#== Global Variables ==#
test_dataset = {'data': [
    {'question': 'What degree programs does the department offer?', 'label': 'Degree Programs'},
    {'question': 'What dual degrees can I pursue?', 'label': 'Degree Programs'},
    {'question': 'What are the various research areas in the Lane Department?', 'label': 'Research Opportunities'},
    {'question': 'What research is done in the biometrics field?', 'label': 'Research Opportunities'},
    {'question': 'What are the student orgs I can join as a LCSEE student?', 'label': 'Clubs and Organizations'},
    {'question': 'What kind of activities do CyberWVU students do?', 'label': 'Clubs and Organizations'},
    {'question': 'What can I do with a computer engineering degree?', 'label': 'Career Opportunities'},
    {'question': 'What can I do with a computer science degree?', 'label': 'Career Opportunities'},
]}

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

def classify_and_extract(args: dict) -> None:
    '''
    use classification method and extraction method to fill the test dataset with
    their predicted class and keywords

    Args:
        args (dict): script arguments (classify_method and extract_method)
    '''
    # load the dataset
    dataset = load_dataset('../dataset.json')

    #-- determine the classification used --#
    if args.classify_method:
        #- Linear Regression -#
        if args.classify_method == 'LR':
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

            # store predicted class in test dataset 
            for i,item in enumerate(test_dataset['data']):
                question = item['question']
                predicted_category = trad.classify_question(question, vectorizer, model, label_encoder, False)
                test_dataset['data'][i]['pred_class'] = predicted_category

        #- Support Vector Machine -#
        elif args.classify_method == 'SVM':
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

            # store predicted class in test dataset 
            for i,item in enumerate(test_dataset['data']):
                question = item['question']
                predicted_category = trad.classify_question(question, vectorizer, model, label_encoder, False)
                test_dataset['data'][i]['pred_class'] = predicted_category

        #- BERT -#
        elif args.classify_method == 'BERT':
            print('Using BERT')

        #- DistilBERT -#
        elif args.classify_method == 'DistilBERT':
            print('Using DistilBERT')

        else:
            print('Error: Classification method not defined')
            quit()
    else: 
        print('Error: Provide classification method')
        quit()

    #-- determine extraction method --#
    # put keywords associated with question into test dataset 
    if args.extract_method:
        #- NER -#
        if args.extract_method == 'NER':
            print('Using NER')

            # load pre-trained NER model
            nlp = spacy.load("en_core_web_sm")

            # extract keywords for each test question 
            for i,question in enumerate(test_dataset['data']):
                kw = trad.extract_keywords_NER(question['question'], nlp)
                test_dataset['data'][i]['keywords'] = kw

        #- TF-IDF -#
        elif args.extract_method == 'TFIDF':
            print('Using TF-IDF')

            # convert questions to TF-IDF features
            vectorizer = TfidfVectorizer(stop_words='english')
            X_tfidf = vectorizer.fit_transform([item['question'] for item in test_dataset['data']])

            # extract keywords for each test question 
            for i,question in enumerate(test_dataset['data']):
                kw = trad.extract_keywords_TFIDF(question['question'], vectorizer)
                test_dataset['data'][i]['keywords'] = kw

        #- Word2Vec -#
        elif args.extract_method == 'vec':
            print('Using Word2Vec')

            # load a pretrained model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # extract keywords for each test question 
            for i,question in enumerate(test_dataset['data']):
                kw_vec = model.encode(question['question'])
                test_dataset['data'][i]['keywords'] = kw_vec
    else: 
        print('Error: Provide extraction method')
        quit()

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

#== Main Execution ==#
def main(args):
    # fill test dataset with predictions using provided classification method and extraction method
    classify_and_extract(args)

    #-- extract the answer from the database --#
    for i,item in enumerate(test_dataset['data']):
        question = item['question']
        pred_class = item['pred_class']
        kw = item['keywords']

        # get filtered data from class
        filtered_data = filter_dataset(pred_class)
        
        # get best answer (based on method)
        if args.retrieve_method == 'EKI':
            answer = retrieve_EKI(filtered_data, kw)

            print(f'Question: {question}\n\tAnswer: {answer}\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Retrieve Step')

    # arguments
    argparser.add_argument('--classify_method', type=str, choices=['LR', 'SVM', 'BERT', 'DistilBERT'], help='Classification method to use')
    argparser.add_argument('--extract_method', type=str, choices=['NER', 'TFIDF', 'vec'], help='Extraction of question information. Value passed determines method used:\nNER - Named Entity Recognition\nTFIDF - TF-IDF vectorization\n\vec - Word2Vec')
    argparser.add_argument('--retrieve_method', type=str, choices=['EKI'], help='Retrieval method to use')

    args = argparser.parse_args()

    main(args)