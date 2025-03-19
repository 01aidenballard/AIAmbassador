'''
Classify-Retrieve-Generate API
Provides classes and methods to use CRG in Python Script

Author: Ian Jackson
Date: 03/18/2025
'''

#== Imports ==#
import os
import sys
import json

from enum import Enum
from typing import Union
from classify.traditional_ML import LogisticRegression, SVM
from sklearn.feature_extraction.text import TfidfVectorizer

#== Enums ==#
class ClassifyMethod(Enum):
    LR = 1
    SVM = 2
    BERT = 3
    DISTILBERT = 4

class ExtractMethod(Enum):
    NER = 1
    TFIDF = 2
    VEC = 3

#== Global Variables ==#


#== Classes ==#
class Dataset():
    '''
    class to represent labeled dataset
    '''
    def __init__(self, path: str):
        '''
        initialize dataset

        Args:
            path (str): path to dataset
        '''
        self.path = path
        
        # load dataset 
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> dict:
        '''
        load dataset with questions, answer, class label

        Returns:
            dict: labeled dataset 
        '''
        # check if path exist
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"[E] Dataset not found at {self.path}")
        
        # load the JSON file
        with open(self.path, 'r') as f:
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
                answer = qa['answer'][0]

                labeled_data['data'].append(
                    {
                        'question': question,
                        'answer': answer,
                        'label': label
                    }
                )

        return labeled_data


class Classify():
    '''
    class to represent classification/extraction step
    '''
    def __init__(self, classify_method: ClassifyMethod = ClassifyMethod.LR, extract_method: ExtractMethod = ExtractMethod.VEC):
        '''
        init instance of classification and extraction method

        Args:
            classify_method (ClassifyMethod, optional): Classification method. Defaults to ClassifyMethod.LR.
            extract_method (ExtractMethod, optional): extraction method. Defaults to ExtractMethod.VEC.
        '''
        self.classify_method = classify_method
        self.extract_method = extract_method

        # init model and vectorizer/tokenizer
        if self.classify_method in [ClassifyMethod.LR, ClassifyMethod.SVM]:
            self.model, self.vectorizer = self._init_trad_model_vectorizer(self.classify_method)

        elif self.classify_method in [ClassifyMethod.BERT, ClassifyMethod.DISTILBERT]:
            self.model, self.tokenizer = self._init_trad_model_tokenizer(self.classify_method)

        else:
            print(f'[E] Invalid classification method {classify_method}')

    def _init_trad_model_vectorizer(self, classify_method: ClassifyMethod) -> Union[LogisticRegression | SVM, TfidfVectorizer]:
        '''
        initialize classification model and tokenizer
        used for traditional ML classification type
        has to be pretrained, wont train

        Args:
            classify_method (ClassifyMethod): classification method (LR or SVM)

        Returns:
            Union[LogisticRegression | SVM, TfidfVectorizer]: tuple of model and tokenizer
                - model, either LR instance or SVM instance
                - TFIDF tokenizer 
        '''

    def _init_trad_model_tokenizer():
        pass


class Retrieve():
    pass

class Generate():
    pass

class CRG():
    '''
    Class to represent CRG method
    '''
    def __init__(self):
        pass

#== Methods ==#
