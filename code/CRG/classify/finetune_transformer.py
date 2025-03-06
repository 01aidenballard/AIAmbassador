'''
CRG - Classify Research
Fine-Tuning a Pretrained Transformer Model

Author: Ian Jackson
Date: 03-05-2025

'''

#== Imports ==#
import os
import json
import time
import argparse
import torch
import spacy

import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader

#== Global Variables ==#
GB_test_dataset = {'data': [
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

#== Main Execution ==#
def main(args):
    # load the dataset
    dataset = load_dataset('../dataset.json')

    # extract force train
    train = False
    if args.force: train = True

    # preprocess the dataset
    train_questions = [item['question'] for item in dataset['data']]
    train_labels = [item['label'] for item in dataset['data']]

    # encode labels to numerical values
    label_encoder = LabelEncoder()
    label_map = label_encoder.fit_transform(train_labels)

    #-- BERT --#
    if args.BERT:
        print('Using BERT for classification')

        # check if model exist
        if not os.path.exists(f"./classify/bert-question-classifier") or train:
            print('Training BERT')

            # load the tokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            # tokenize the training & test data
            train_encodings = tokenizer(train_questions, truncation=True, padding=True, max_length=512)

            # convert to HuggingFace dataset format
            train_dataset = Dataset.from_dict({
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask'],
                'labels': label_map
            })

            # 80/20 test eval split
            train_data = train_dataset.train_test_split(test_size=0.8, seed=42)['train']
            eval_data = train_dataset.train_test_split(test_size=0.2, seed=42)['test']

            # load pretrained BERT model
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

            # define training args
            training_args = TrainingArguments(
                output_dir=f"./classify/BERT_results",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir=f"./classify/logs/BERT",
                logging_steps=10,
            )

            # use HF trainer API
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
            )

            # Train the model
            trainer.train()

            # save the model
            model.save_pretrained(f"./classify/bert-question-classifier")
            tokenizer.save_pretrained(f"./classify/bert-question-classifier")
        else:
            print('Loading BERT model')
            model = BertForSequenceClassification.from_pretrained(f"./classify/bert-question-classifier")
            tokenizer = BertTokenizer.from_pretrained(f"./classify/bert-question-classifier")

        # evaluate BERT on test dataset
        model.eval()

        true_labels = []
        predicted_labels = []
        response_times = []

        for item in GB_test_dataset['data']:
            question = item['question']
            true_label = item['label']

            # tokenize question
            inputs = tokenizer(question, return_tensors="pt")

            start_time = time.time()

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()

            end_time = time.time() 
            response_time = end_time - start_time
            response_times.append(response_time)

            # map to label
            category = label_encoder.inverse_transform([prediction])[0]
            print(f"Question: {question}\n\tPredicted Category: {category}")

            true_labels.append(true_label)
            predicted_labels.append(category)

        # Compute Evaluation Metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average="weighted")
        avg_response_time = sum(response_times) / len(response_times)

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Average Response Time: {avg_response_time:.2f} ms")

    #-- DistilBERT --#
    if args.DistilBERT:
        print('Using DistilBERT for classification')

    #-- T5 --#
    if args.T5:
        print('Using T5 for classification')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Traditional ML Classifier')

    # arguments
    argparser.add_argument('--BERT', action='store_true', help='Use BERT for classification')
    argparser.add_argument('--DistilBERT', action='store_true', help='Use DistilBERT for classification')
    argparser.add_argument('--T5', action='store_true', help='Use T5 for classification')
    argparser.add_argument('--force', action='store_true', help='force train selected model')

    args = argparser.parse_args()

    main(args)