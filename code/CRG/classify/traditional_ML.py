'''
CRG - Classify Research
Traditional ML Implementation

Author: Ian Jackson
Date: 03-02-2025

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
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

#== Global Variables ==#
test_dataset = {}


#== Classes ==#
class LogisticRegression(nn.Module):
    '''
    Logistic regression model for question classification
    '''
    def __init__(self, input_dim: int, num_classes: int):
        '''
        initialize the LR model

        Args:
            input_dim (int): dimension of the input
            num_classes (int): number of classes
        '''
        super(LogisticRegression, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        feed forward pass of the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        '''
        x = self.bn(x)
        x = self.dropout(self.linear(x))
        return x

class SVM(nn.Module):
    '''
    SVM model for question classification
    '''
    def __init__(self, input_dim: int, num_classes: int):
        '''
        initialize the SVM model

        Args:
            input_dim (int): dimension of input
            num_classes (int): number of classes
        '''
        super(SVM, self).__init__()
        # self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        feed forward pass of the model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        '''
        # x = self.bn(x)
        return self.dropout(self.fc(x))

#== Methods ==#
def load_testset(path: str) -> dict:
    '''
    load the custom test dataset and label the questions for classification

    Args:
        path (str): path to custom test dataset

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

    for ql in data: # iterate through each question/label
        # extract the label and q data
        label = ql['label']
        question = ql['question']

        labeled_data['data'].append(
            {
                'question': question,
                'label': label
            }
        )

    return labeled_data


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

# def hinge_loss(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
#     '''
#     hinge loss for SVM

#     Args:
#         output (torch.Tensor): output from model
#         target (torch.Tensor): actual labels to compare against
#         num_classes (int): number of classes

#     Returns:
#         float: hinge loss
#     '''
#     # convert labels to one-hot encoding
#     target_one_hot = torch.eye(num_classes)[target] 

#     # calculate the hinge loss
#     margin = 1 - output * target_one_hot
#     loss = torch.mean(torch.clamp(margin, min=0))  # max(0, 1 - y*f(x))
#     return loss

def hinge_loss(output: torch.Tensor, target: torch.Tensor, num_classes: int, margin: float = 1.0) -> torch.Tensor:
    """
    Multi-class hinge loss for SVM.

    Args:
        output (torch.Tensor): shape (batch_size, num_classes), raw model outputs
        target (torch.Tensor): shape (batch_size,), correct class indices
        num_classes (int): total number of classes
        margin (float): margin between correct and incorrect scores

    Returns:
        torch.Tensor: scalar loss
    """
    batch_size = output.size(0)

    # Gather the correct class scores using index selection
    correct_class_scores = output[torch.arange(batch_size), target].unsqueeze(1)  # shape (batch_size, 1)

    # Compute margin loss for all classes
    margins = torch.clamp(output - correct_class_scores + margin, min=0)  # shape (batch_size, num_classes)

    # Do not include correct class in the loss
    margins[torch.arange(batch_size), target] = 0

    # Return mean loss
    loss = torch.mean(torch.sum(margins, dim=1))
    return loss


def classify_question(question: str, vectorizer: TfidfVectorizer, model: LogisticRegression | SVM, label_encoder: LabelEncoder, show_output: bool = True) -> str:
    '''
    classify a question into category using trained model
    can use either LR or SVM since the both use TF-IDF features

    Args:
        question (str): question to be classified
        vectorizer (TfidfVectorizer): TD-IDF vectorizer
        model (LogisticRegression | SVM): model to use for classification
        label_encoder (LabelEncoder): label encoder to decode predicted label

    Returns:
        str: predicted class
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
    if show_output: print(f"Question: {question}\n\tPredicted Category: {category} | Response Time: {response_time:.2f} ms\n")
    
    return category

def extract_keywords_NER(question: str, nlp) -> list:
    '''
    extract named entities and keywords from question

    Args:
        question (str): question of interest
        nlp (_type_): spacy model

    Returns:
        list: extracted entities and keywords
    '''
    doc = nlp(question)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    return keywords + entities

def extract_keywords_TFIDF(question: str, vectorizer: TfidfVectorizer) -> list:
    '''
    extract keywords from question using TF-IDF vectorization

    Args:
        question (str): question of interest
        vectorizer (TfidfVectorizer): tf-idf vectorizer

    Returns:
        list: extracted keywords
    '''
    response = vectorizer.transform([question])
    feature_names = vectorizer.get_feature_names_out()
    keywords = [feature_names[i] for i in response.indices]
    return keywords

#== Main Execution ==#
def main(args):
    # extract force arg
    force = False
    if args.force: force = True

    # load the dataset
    dataset = load_dataset('../dataset.json')

    # load the test dataset
    global test_dataset
    test_dataset = load_testset('../test_dataset.json')

    #-- Logistic Regression --#
    if args.LR:
        print("Running Logistic Regression")

        # extract the questions and labels
        questions = [item['question'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]

        # convert text into TF-IDF features
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
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

        #- Train the Model -#
        if not os.path.exists("classify/lr_c_model.pth") or force:
            num_epochs = 5000
            loss_val = []

            for epoch in range(num_epochs):
                model.train()

                # forward pass
                outputs = model(X_tensor)
                loss = criterion(outputs, Y_tensor)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store loss
                loss_val.append(loss.item())
                scheduler.step()

                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            torch.save(model.state_dict(), "classify/lr_c_model.pth")
            print("Model saved successfully")
            model.eval()

            # plot training loss curve
            plt.figure(figsize=(8,6))
            plt.plot(range(1, num_epochs+1), loss_val)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.grid()
            plt.savefig('classify/LR_train_curve.png')
        else:
            print("Loading trained model")
            model.load_state_dict(torch.load("classify/lr_c_model.pth"))
            model.eval()

        #- Run the model -#
        print("Evaluating model")
        
        response_times = []
        true_labels = []
        predicted_labels = []

        for item in test_dataset['data']:
            question = item['question']
            true_label = item['label']
            
            start_time = time.time()
            predicted_category = classify_question(question, vectorizer, model, label_encoder)
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

    #-- Support Vector Machine --#
    elif args.SVM:
        print("Running Support Vector Machine")

        # extract the questions and labels
        questions = [item['question'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]

        # convert text into TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000)
        X_tfidf = vectorizer.fit_transform(questions).toarray()

        # encode labels to numerical values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)

        # convert data into PyTorch Tensors
        X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
        Y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # initialize the model
        input_dim = X_tensor.shape[1]
        num_classes = len(set(labels))
        model = SVM(input_dim, num_classes)

        # loss is hinge loss, use Adam optimizer
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        #- Train the Model -#
        if not os.path.exists("classify/svm_c_model.pth") or force:
            num_epochs = 1500
            batch_size = 1470
            loss_val = []

            for epoch in range(num_epochs):
                model.train()
                total_loss = 0

                for i in range(0, len(X_tensor), batch_size):
                    # get batch
                    X_batch = X_tensor[i:i+batch_size]
                    y_batch = Y_tensor[i:i+batch_size]

                    # forward Pass
                    optimizer.zero_grad() 
                    outputs = model(X_batch)
                    loss = hinge_loss(outputs, y_batch, num_classes)

                    # backward Pass
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # store loss
                loss_val.append(total_loss)

                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

            torch.save(model.state_dict(), "classify/svm_c_model.pth")
            print("Model saved successfully")
            model.eval()

            # plot training loss curve
            plt.figure(figsize=(8,6))
            plt.plot(range(1, num_epochs+1), loss_val)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.grid()
            plt.savefig('classify/SVM_train_curve.png')
        else:
            print("Loading trained model")
            model.load_state_dict(torch.load("classify/svm_c_model.pth"))
            model.eval()

        #- Run the model -#
        print("Evaluating model")

        response_times = []
        true_labels = []
        predicted_labels = []

        for item in test_dataset['data']:
            question = item['question']
            true_label = item['label']
            
            start_time = time.time()
            predicted_category = classify_question(question, vectorizer, model, label_encoder)
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

    #-- Extract Information --#
    elif args.extract:
        #- NER -#
        if args.extract == 'NER':
            # load pre-trained NER model
            nlp = spacy.load("en_core_web_sm")

            # extract keywords for each test question 
            for question in test_dataset['data']:
                kw = extract_keywords_NER(question['question'], nlp)
                print(f"Question: {question['question']}")
                print(f"\tKeywords: {kw}")
                print()

        #- TF-IDF -#
        elif args.extract == 'TFIDF':
            # convert questions to TF-IDF features
            vectorizer = TfidfVectorizer(stop_words='english')
            X_tfidf = vectorizer.fit_transform([item['question'] for item in test_dataset['data']])

            # extract keywords for each test question 
            for question in test_dataset['data']:
                kw = extract_keywords_TFIDF(question['question'], vectorizer)
                print(f"Question: {question['question']}")
                print(f"\tKeywords: {kw}")
                print()

        #- Word2Vec -#
        elif args.extract == 'vec':
            # load a pretrained model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # extract keywords for each test question 
            for question in test_dataset['data']:
                kw_vec = model.encode(question['question'])
                print(f"Question: {question['question']}")
                print(f"\tVector length: {len(kw_vec)}\tFirst 3 elements: {kw_vec[:3]}")
                print()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Traditional ML Classifier')

    argparser.add_argument('--LR', action='store_true', help='Use Logistic Regression')
    argparser.add_argument('--SVM', action='store_true', help='Use Support Vector Machine')
    argparser.add_argument('--force', action='store_true', help='Force training')
    argparser.add_argument('--extract', type=str, choices=['NER', 'TFIDF', 'DP', 'vec'], help='Run extraction of question information. Value passed determines method used:\nNER - Named Entity Recognition\nTFIDF - TF-IDF vectorization\n\vec - Word2Vec')

    args = argparser.parse_args()

    main(args)
