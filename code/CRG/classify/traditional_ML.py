'''
CRG - Classify Research
Traditional ML Implementation

Author: Ian Jackson
Date: 03-02-2025

'''

#== Imports ==#
import os
import json
import torch
import argparse
import time

import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

#== Global Variables ==#


#== Classes ==#
class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

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

def classify_question_LR(question: str, vectorizer: TfidfVectorizer, model: LogisticRegression, label_encoder) -> str:
    '''
    classify question using LR model

    Args:
        question (str): question to classify
        vectorizer (TfidfVectorizer): TF-IDF vectorizer
        model (LogisticRegression): trained model
    Returns:
        str: class
    '''
    # measure response time
    start_time = time.time()

    # convert question into TF-IDF features
    question_tfidf = vectorizer.transform([question]).toarray()
    question_tensor = torch.tensor(question_tfidf, dtype=torch.float32)

    # get model prediction
    model.eval()
    with torch.no_grad():
        output = model(question_tensor)
        predicted_label = torch.argmax(output, dim=1).item()

    # convert label back to category name
    category = label_encoder.inverse_transform([predicted_label])[0]

    response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Predicted Category: {category} | Response Time: {response_time:.2f} ms")
    
    return category

#== Main Execution ==#
def main(args):
    # extract force arg
    force = False
    if args.force: force = True

    # load the dataset
    dataset = load_dataset('../dataset.json')

    #-- Logistic Regression --#
    if args.LR:
        # extract the questions and labels
        questions = [item['question'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]

        # convert test into TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000)
        X_tfidf = vectorizer.fit_transform(questions).toarray()

        # encode labels to numerical values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        # print("Label Mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

        # convert data into PyTorch Tensors
        X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
        Y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # initialize the model
        input_dim = X_tensor.shape[1]
        num_classes = len(set(labels))
        model = LogisticRegression(input_dim, num_classes)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #- Train the Model -#
        if not os.path.exists("classify/lr_c_model.pth") or force:
            num_epochs = 100

            for epoch in range(num_epochs):
                model.train()

                # forward pass
                outputs = model(X_tensor)
                loss = criterion(outputs, Y_tensor)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            torch.save(model.state_dict(), "classify/lr_c_model.pth")
            print("Model saved successfully")
            model.eval()
        else:
            print("Loading trained model")
            model.load_state_dict(torch.load("classify/lr_c_model.pth"))
            model.eval()

        #- Run the model -#
        print("Evaluating model")
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
        
        response_times = []
        true_labels = []
        predicted_labels = []

        for item in test_dataset['data']:
            question = item['question']
            true_label = item['label']
            
            start_time = time.time()
            predicted_category = classify_question_LR(question, vectorizer, model, label_encoder)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            response_times.append(response_time)
            true_labels.append(true_label)
            predicted_labels.append(predicted_category)

        # Compute Evaluation Metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average="weighted")
        avg_response_time = sum(response_times) / len(response_times)

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Average Response Time: {avg_response_time:.2f} ms")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Traditional ML Classifier')

    argparser.add_argument('--LR', action='store_true', help='Use Logistic Regression')
    argparser.add_argument('--force', action='store_true', help='Force training')

    args = argparser.parse_args()

    main(args)