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
import torch
import spacy
import random

import numpy as np 
import torch.nn as nn

from enum import Enum
from typing import Union
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from classify.traditional_ML import LogisticRegression, SVM
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


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

class RetrieveMethod(Enum):
    EKI = 1
    Jaccard = 2
    JEKI = 3
    CSS_TFIDF = 4
    CSS_VEC = 5

class GenerateMethod(Enum):
    FLAN_T5 = 1
    TINY_LLAMA = 2
    CONTEXT_ONLY = 3


#== Global Variables ==#
MODEL_LR_PTH = 'classify/lr_c_model.pth'
MODEL_SVM_PTH = 'classify/svm_c_model.pth'
MODEL_BERT_PTH = 'classify/bert-question-classifier'
MODEL_DISTILBERT_PTH = 'classify/distilbert-question-classifier'

MODEL_SPACY = None
MODEL_TFIDF_VEC = None
MODEL_W2V = None

#== Model Classes ==#
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

#== CRG Classes ==#
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
        self.filtered_dataset = None

        # get classes
        self.questions = [item['question'] for item in self.dataset['data']]
        self.labels = [item['label'] for item in self.dataset['data']]
        self.n_classes = len(set(self.labels))

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
    def __init__(self, 
                 dataset: Dataset,
                 classify_method: ClassifyMethod = ClassifyMethod.LR, 
                 extract_method: ExtractMethod = ExtractMethod.VEC):
        '''
        init instance of classification and extraction method

        Args:
            dataset (Dataset): Dataset instance
            classify_method (ClassifyMethod, optional): Classification method. Defaults to ClassifyMethod.LR.
            extract_method (ExtractMethod, optional): extraction method. Defaults to ExtractMethod.VEC.
        '''
        self.dataset = dataset
        self.classify_method = classify_method
        self.extract_method = extract_method

        # if classify model is None, instance is used just for extraction
        if self.classify_method != None:

            # init model and vectorizer/tokenizer
            if self.classify_method in [ClassifyMethod.LR, ClassifyMethod.SVM]:
                self.model, self.vectorizer, self.label_encoder = self._init_trad_model_vectorizer(self.classify_method)
                self.tokenizer = None

            elif self.classify_method in [ClassifyMethod.BERT, ClassifyMethod.DISTILBERT]:
                self.model, self.tokenizer, self.label_encoder = self._init_trad_model_tokenizer(self.classify_method)
                self.vectorizer = None

            else:
                print(f'[E] Invalid classification method {classify_method}')
                quit()

    def _init_trad_model_vectorizer(self, classify_method: ClassifyMethod) -> Union[LogisticRegression | SVM, TfidfVectorizer, LabelEncoder]:
        '''
        initialize classification model and vectorizer
        used for traditional ML classification type
        has to be pretrained, wont train

        Args:
            classify_method (ClassifyMethod): classification method (LR or SVM)

        Returns:
            Union[LogisticRegression | SVM, TfidfVectorizer, LabelEncoder]: tuple of model, vectorizer, and label encoder
                - model, either LR instance or SVM instance
                - TFIDF vectorizer 
                - label encoder
        '''
        # initialize vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)

        # initialize and fit label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(self.dataset.labels)

        # get input dimensions using vectorizer
        X_tfidf = vectorizer.fit_transform(self.dataset.questions).toarray()
        X_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
        input_dim = X_tensor.shape[1]

        # init model and load pretrained
        if classify_method == ClassifyMethod.LR:
            model = LogisticRegression(input_dim, self.dataset.n_classes)

            if os.path.exists(MODEL_LR_PTH):
                model.load_state_dict(torch.load(MODEL_LR_PTH))
            else:
                print(f'[E] Pretrained LR model not found at {MODEL_LR_PTH}')
                quit()

        elif classify_method == ClassifyMethod.SVM:
            model = SVM(input_dim, self.dataset.n_classes)

            if os.path.exists(MODEL_SVM_PTH):
                model.load_state_dict(torch.load(MODEL_SVM_PTH))
            else:
                print(f'[E] Pretrained SVM model not found at {MODEL_SVM_PTH}')
                quit()

        # set model to evaluation mode
        model.eval()

        return model, vectorizer, label_encoder

    def _init_trad_model_tokenizer(self, classify_method: ClassifyMethod) -> Union[BertForSequenceClassification | DistilBertForSequenceClassification, BertTokenizer | DistilBertTokenizer, LabelEncoder]:
        '''
        initialize classification model and tokenizer
        used for finetune transformer classification type
        has to be pretrained, wont train

        Args:
            classify_method (ClassifyMethod): classification method (BERT or DistilBERT)

        Returns:
            Union[BertForSequenceClassification | DistilBertForSequenceClassification, BertTokenizer | DistilBertTokenizer, LabelEncoder]:
                - model, either BERT instance or DistilBERT instance
                - BERT or DistilBERT vectorizer 
                - label encoder
        '''
        # initialize and fit label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(self.dataset.labels)

        # load pretrained model and tokenizer
        if classify_method == ClassifyMethod.BERT:
            # check if exist
            if os.path.exists(MODEL_BERT_PTH):
                model = BertForSequenceClassification.from_pretrained(MODEL_BERT_PTH)
                tokenizer = BertTokenizer.from_pretrained(MODEL_BERT_PTH)
            else:
                print(f'[E] Pretrained BERT model not found at {MODEL_BERT_PTH}')
                quit()

        elif classify_method == ClassifyMethod.DISTILBERT:
            # check if exist
            if os.path.exists(MODEL_DISTILBERT_PTH):
                model = DistilBertForSequenceClassification.from_pretrained(MODEL_DISTILBERT_PTH)
                tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DISTILBERT_PTH)
            else:
                print(f'[E] Pretrained DistilBERT model not found at {MODEL_DISTILBERT_PTH}')
                quit()
        
        # put model in eval mode
        model.eval()

        return model, tokenizer, label_encoder

    def classify_question(self, question: str) -> str:
        '''
        classifies a question using the chosen classification model

        Args:
            question (str): question to classify

        Returns:
            str: question class
        '''
        # determine classification method 
        # LR or SVM
        if self.classify_method in [ClassifyMethod.LR, ClassifyMethod.SVM]:
            # convert question to TFIDF tensor
            question_tfidf = self.vectorizer.transform([question]).toarray()
            question_tensor = torch.tensor(question_tfidf, dtype=torch.float32)

            # get prediction
            with torch.no_grad():
                output = self.model(question_tensor)
                pred_label = torch.argmax(output, dim=1).item()

            # convert label back to category name
            label = self.label_encoder.inverse_transform([pred_label])[0]

        # BERT or DistilBERT
        elif self.classify_method in [ClassifyMethod.BERT, ClassifyMethod.DISTILBERT]:
            # transform question using tokenizer
            inputs = self.tokenizer(question, return_tensors='pt')

            # get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred_label = torch.argmax(outputs.logits, dim=-1).item()

            # map to label
            label = self.label_encoder.inverse_transform([pred_label])[0]

        return label

    def extract_info(self, question: str) -> list | torch.Tensor:
        '''
        extracts information from question for retrieval

        Args:
            question (str): question to extract from

        Returns:
            list | torch.Tensor: list of keywords or vector embedding
        '''
        # determine the extraction method
        # NER
        if self.extract_method == ExtractMethod.NER:
            # load spacy model
            nlp = spacy.load("en_core_web_sm")
            global MODEL_SPACY
            MODEL_SPACY = nlp

            # extract keywords
            doc = nlp(question)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

            info = entities + keywords

        # TFIDF
        elif self.extract_method == ExtractMethod.TFIDF:
            # init TFIDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            vectorizer.fit_transform(self.dataset.questions)

            global MODEL_TFIDF_VEC
            MODEL_TFIDF_VEC = vectorizer

            # extract keywords
            response = vectorizer.transform([question])
            feature_names = vectorizer.get_feature_names_out()
            info = [feature_names[i] for i in response.indices]

        # VEC
        elif self.extract_method == ExtractMethod.VEC:
            # load a pretrained model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            global MODEL_W2V
            MODEL_W2V = model

            info = model.encode(question)

        else:
            print(f'[E] Invalid extraction method {self.extract_method}')
            quit()

        return info

class Retrieve():
    '''
    class to represent retrieval step
    '''
    def __init__(self, 
                 dataset: Dataset, 
                 retrieve_method: RetrieveMethod = RetrieveMethod.CSS_VEC, 
                 extract_method: ExtractMethod = None,
                 lambd_1: float = 0.5, lambd_2: float = 0.5):
        '''
        init instance of retrieval step

        Args:
            dataset (Dataset): dataset instance
            retrieve_method (RetrieveMethod, optional): method of retrieval. Defaults to RetrieveMethod.CSS_VEC.
            extract_method (ExtractMethod): method of extraction from classification step (needed in Jaccard)
            lambd_1 (float): weight for EKI score (default 0.5)
            lambd_2 (float): weight for Jaccard score (default 0.5)
        '''
        self.dataset = dataset
        self.retrieve_method = retrieve_method
        self.extract_method = extract_method
        self.lmbda_1 = lambd_1
        self.lmbda_2 = lambd_2

    def retrieve_answer(self, question: str, question_info: list | torch.Tensor) -> str:
        '''
        retrieve answer from database using specified method
        IMPORTANT: for efficient use, set dataset.filtered_dataset BEFORE retrieving

        Returns:
            str: best answer
        '''
        # determine retrieval method
        # EKI
        if self.retrieve_method == RetrieveMethod.EKI:
            # tag each question in filtered dataset with score
            for i,qa in enumerate(self.dataset.filtered_dataset['data']):
                question = qa['question']
                
                score = 0

                # for each keyword, see if its in the question
                for kw in question_info:
                    # keywords can be either strings or tuples
                    if type(kw) is str:
                        if kw in question: score += 1
                    elif type(kw) is tuple:
                        x,y = kw
                        if x in question: score += 1
                        if y in question: score += 1

                self.dataset.filtered_dataset['data'][i]['score'] = score

            # get the highest scoring answers
            max_score = max(item['score'] for item in self.dataset.filtered_dataset['data'])
            best_ans = [item['answer'] for item in self.dataset.filtered_dataset['data'] if item['score'] == max_score]
            
            # flatten 
            # best_ans = [item[0] for item in best_ans]

            # if one answer, use that; otherwise choose randomly
            if len(best_ans) == 1:
                best_ans = best_ans[0]
            else:
                best_ans = best_ans[random.randint(0, len(best_ans) - 1)]

        # Jaccard
        # TODO: Pre-Score Shtuff
        elif self.retrieve_method == RetrieveMethod.Jaccard:
            classification = Classify(self.dataset, None, self.extract_method)

            for i,qa in enumerate(self.dataset.filtered_dataset['data']):
                question = qa['question']
                question_kw = None

                # to use extraction code, init classification instance
                question_kw = classification.extract_info(question)

                # flatten with possibility of tuples
                question_info = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_info))
                question_kw = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_kw))

                # compute Jaccard similarity score
                intersection = set(question_info) & set(question_kw)
                union = set(question_info) | set(question_kw)
                score = len(intersection) / len(union) if union else 0

                # store score in dataset
                self.dataset.filtered_dataset['data'][i]['score'] = score

            # get the highest scoring answers
            max_score = max(item['score'] for item in self.dataset.filtered_dataset['data'])
            best_ans = [item['answer'] for item in self.dataset.filtered_dataset['data'] if item['score'] == max_score]
            
            # flatten 
            # best_ans = [item[0] for item in best_ans]

            # if one answer, use that; otherwise choose randomly
            if len(best_ans) == 1:
                best_ans = best_ans[0]
            else:
                best_ans = best_ans[random.randint(0, len(best_ans) - 1)]

        # JEKI
        elif self.retrieve_method == RetrieveMethod.JEKI:
            classification = Classify(self.dataset, None, self.extract_method)

            for i,qa in enumerate(self.dataset.filtered_dataset['data']):
                question = qa['question']
                question_kw = None

                # to use extraction code, init classification instance
                question_kw = classification.extract_info(question)

                # flatten with possibility of tuples
                question_info = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_info))
                question_kw = list(chain.from_iterable(item if isinstance(item, tuple) else (item,) for item in question_kw))

                # compute Jaccard similarity score
                intersection = set(question_info) & set(question_kw)
                union = set(question_info) | set(question_kw)
                jaccard_score = len(intersection) / len(union) if union else 0

                # get EKI score
                EKI_score = 0

                # for each keyword, see if its in the question
                for kw in question_info:
                    # keywords can be either strings or tuples
                    if type(kw) is str:
                        if kw in question: EKI_score += 1
                    elif type(kw) is tuple:
                        x,y = kw
                        if x in question: EKI_score += 1
                        if y in question: EKI_score += 1

                # store score in dataset
                self.dataset.filtered_dataset['data'][i]['score'] = self.lmbda_1 * EKI_score + self.lmbda_2 * jaccard_score

            # get the highest scoring answers
            max_score = max(item['score'] for item in self.dataset.filtered_dataset['data'])
            best_ans = [item['answer'] for item in self.dataset.filtered_dataset['data'] if item['score'] == max_score]
            
            # flatten 
            # best_ans = [item[0] for item in best_ans]

            # if one answer, use that; otherwise choose randomly
            if len(best_ans) == 1:
                best_ans = best_ans[0]
            else:
                best_ans = best_ans[random.randint(0, len(best_ans) - 1)]

        # CSS-TDIDF
        elif self.retrieve_method == RetrieveMethod.CSS_TFIDF:
            for i,qa in enumerate(self.dataset.filtered_dataset['data']):
                db_question = qa['question']

                tdidf_matrix = MODEL_TFIDF_VEC.transform([question, db_question])
                cosine_sim_matrix = cosine_similarity(tdidf_matrix, tdidf_matrix)

                # extract score
                css = cosine_sim_matrix[0,1]

                # store score in dataset
                self.dataset.filtered_dataset['data'][i]['score'] = css

            # get the highest scoring answers
            max_score = max(item['score'] for item in self.dataset.filtered_dataset['data'])
            best_ans = [item['answer'] for item in self.dataset.filtered_dataset['data'] if item['score'] == max_score]

            # if one answer, use that; otherwise choose randomly
            if len(best_ans) == 1:
                best_ans = best_ans[0]
            else:
                best_ans = best_ans[random.randint(0, len(best_ans) - 1)]

        # CSS-vec
        # TODO: Pre-Vectorize
        elif self.retrieve_method == RetrieveMethod.CSS_VEC:
            # get vector embedding of asked question
            ask_question_vec = MODEL_W2V.encode(question)
            ask_question_vec = np.array(ask_question_vec).reshape(1, -1)
            
            for i,qa in enumerate(self.dataset.filtered_dataset['data']):
                # get vector embedding of database question
                db_question = qa['question']
                db_question_vec = MODEL_W2V.encode(db_question)
                db_question_vec = np.array(db_question_vec).reshape(1, -1)

                # compute similarity
                css = cosine_similarity(ask_question_vec, db_question_vec)[0, 0]

                # store score in dataset
                self.dataset.filtered_dataset['data'][i]['score'] = css

            # get the highest scoring answers
            max_score = max(item['score'] for item in self.dataset.filtered_dataset['data'])
            best_ans = [item['answer'] for item in self.dataset.filtered_dataset['data'] if item['score'] == max_score]

            # if one answer, use that; otherwise choose randomly
            if len(best_ans) == 1:
                best_ans = best_ans[0]
            else:
                best_ans = best_ans[random.randint(0, len(best_ans) - 1)]

        else:
            print(f'[E] Invalid retrieval method {self.retrieve_method}')
            quit()

        return best_ans

class Generate():
        '''
        Class to represent the generation step using models from generate.py
        '''
        def __init__(self, method: GenerateMethod = GenerateMethod.FLAN_T5):
            '''
            Initialize the generation model

            Args:
                method (GenerateMethod): Enum to select the generation model
            '''
            self.method = method
            if self.method == GenerateMethod.FLAN_T5:
                model_name = "Flan-T5"
            elif self.method == GenerateMethod.TINY_LLAMA:
                model_name = "TinyLlama"
            elif self.method == GenerateMethod.CONTEXT_ONLY:
                model_name = "Context Only"
            else:
                raise ValueError(f"Unknown GenerateMethod: {self.method}")
            self.model = model_name

        def generate_answer(self, question: str, context: str = "") -> str:
            '''
            Generate an answer using the selected generation model

            Args:
                question (str): The input question
                context (str, optional): Additional context or retrieved answer

            Returns:
                str: Generated answer
            '''

            if self.model == "Flan-T5":

                # system prompt for Flan-T5
                system_prompt = """
                    Persona: You are "Lain," a friendly and enthusiastic university tour guide. Your audience is a group of prospective high school students and their families. Your tone should be welcoming, helpful, and engaging.

                    Core Task: Your primary goal is to take a piece of factual information (the "Retreived Answer") and rephrase it into a natural, conversational response("Generated Response") to a "User Question."

                    Instructions:
                    1.  Natural Language: Transform the provided "Retrieved Answer" from a factual statement into a flowing, easy-to-understand sentence or two. Imagine you are speaking directly to someone on a campus tour.
                    2.  Strict Information Adherence: You MUST only use the information provided in the "Retrieved Answer." Do not add any new facts, statistics, or details, even if they seem relevant. Do not hallucinate. Be concise, but thorough with the specifics.
                    3.  No Meta-Commentary: Do not mention that you have been "given" or "provided" with information. The response should be seamless.
                    4.  Engage with a Question: After providing the answer, always ask a relevant, open-ended follow-up question to encourage further conversation.
                    5.  Structure: The final output should only be the conversational reply from Lain.

                    Example of your task:

                    User Question: "What's the student-to-faculty ratio?"
                    Retrieved Answer: "The student-to-faculty ratio is 15 to 1."

                    Generated Response:
                    "That's a great question! We have a student-to-faculty ratio of 15 to 1, which means our professors get to know their students really well. Are you interested in any particular academic departments?"

                    Now, use the following information to answer the user's question:
                    """

                # put together input text
                input_text = (
                    f"{system_prompt}"
                    f"User Question: {question}\n"
                    f"Retrieved Answer: {context}\n"
                    f"Generated Response:"
                )

                
                # load model
                name = 'google/flan-t5-small'
                tokenizer = T5Tokenizer.from_pretrained(name, legacy=False)
                model = T5ForConditionalGeneration.from_pretrained(name)

                # tokenize input
                inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

                # generate response
                output_tokens = model.generate(
                    **inputs,
                    max_length=256,
                    do_sample=True,  # Enables creative responses
                    temperature=0.7,  # Introduces variety
                    top_p=0.9,  # Ensures diverse and high-quality generation
                    repetition_penalty=1.2,  # Prevents repeating phrases
                    num_return_sequences=1,  # Single response
                    eos_token_id=tokenizer.eos_token_id  # Ensures proper sentence ending
                )

                # decode the generated response
                response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

                return response
            
            elif self.model == "TinyLlama":

                # system prompt for TinyLlama
                system_prompt = """
                    You are a friendly and engaging tour guide for West Virginia University.
                    Your role is to provide clear, conversational, and helpful responses based on the given information.

                    - Rephrase the provided information in a natural, engaging way.
                    - Do NOT mention that the information was 'given' or 'provided'.
                    - Do NOT hallucinate or make up information, only use what is given.
                    - Keep responses concise.
                    """

                # user prompt for TinyLlama
                user_prompt = f"Answer the Users Question: {question} with this context: {context}"

                # put together input text
                input_text = (
                    f"User Question: {question}\n"
                    f"Retrieved Answer: {context}\n"
                    f"Generated Response:"
                )
                
                # load model
                model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                tokenizer = AutoTokenizer.from_pretrained(model)
                model = AutoModelForCausalLM.from_pretrained(model)

                messages = [
                    {"role": "system", "content": system_prompt},  
                    {"role": "user", "content": user_prompt},  
                ]

                # tokenize input
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
                input_ids = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                # inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                input_length = input_ids.shape[1]  # Get the length of the input tokens

                # Generate response
                with torch.no_grad():
                    output_tokens = model.generate(
                        # **inputs,
                        input_ids=input_ids,
                        max_length=512,
                        repetition_penalty=1.2,  
                        num_return_sequences=1,  
                        eos_token_id=tokenizer.eos_token_id 
                    )

                # Decode the generated response
                response = tokenizer.decode(output_tokens[0, input_length:], skip_special_tokens=True)
                
                return response
            elif self.model == "Context Only":
                # if no generation model is selected, return the context
                return context
            else:
                raise ValueError(f"Unknown model: {self.model}")
        


class CRG():
    '''
    Class to represent CRG method
    '''
    def __init__(self,
                 dataset_path: str,
                 classify_method: ClassifyMethod = ClassifyMethod.LR, 
                 extract_method: ExtractMethod = ExtractMethod.VEC,
                 retrieve_method: RetrieveMethod = RetrieveMethod.CSS_VEC,
                 generate_method: GenerateMethod = GenerateMethod.FLAN_T5,
                 print_info: bool = False):
        # DOCUMENT: CRG initialization

        self.print_info = print_info

        # initialize dataset
        self.dataset = Dataset(dataset_path)
        if self.print_info: print('✓ Dataset initialized')

        # initialize the classes for each step
        self.classify = Classify(self.dataset, classify_method, extract_method)
        if self.print_info: print('✓ Classification model initialized')

        self.retrieve = Retrieve(self.dataset, retrieve_method, extract_method)
        if self.print_info: print('✓ Retrieval model initialized')

        self.generate = Generate(generate_method)
        if self.print_info: print('✓ Generation model initialized')

    def answer_question(self, question: str) -> str:
        '''
        Uses CRG flow to answer a given question

        Args:
            question (str): question to be asked about LCSEE

        Returns:
            str: Best answer
        '''
        # classify and extract question
        question_class = self.classify.classify_question(question)
        question_info = self.classify.extract_info(question)

        # filter dataset and store in dataset instance
        self.dataset.filtered_dataset = filter_dataset(self.dataset.dataset, question_class)

        # retrieve best answer
        pred_answer = self.retrieve.retrieve_answer(question, question_info)
    
        # generation step
        gen_answer = self.generate.generate_answer(question, pred_answer)

        return gen_answer

#== Methods ==#
def filter_dataset(dataset: dict, label: str) -> dict:
    '''
    filter the dataset to only include QA of one label

    Args:
        dataset (dict): dataset to filter
        label (str): label/class to filter by

    Returns:
        dict: filtered dataset
    '''
    filtered_data = {'data': []}

    for data in dataset['data']:
        if data['label'] == label:
            filtered_data['data'].append(data)

    return filtered_data

def cache_vectors(self, force: bool = False):
        '''
        cache vector embeddings for all questions in dataset
        '''
        # check if path exist
        if not os.path.exists():
            raise FileNotFoundError(f"[E] Dataset not found at {self.path}")
        
        # load the JSON file
        with open(self.path, 'r') as f:
            data = json.load(f)

        data = data['data']

        # iterate through each label
        vector_cache = {'data': []}

        for vector in data:
            # extract the label and qa data
            label = vector['title']
            qas = vector['paragraphs'][0]['qas']

            # extract each question, adding it to the labeled dataset
            for qa in qas:
                question = qa['question']
                answer = qa['answer'][0]

                vector_cache['data'].append(
                    {
                        'question': question,
                        'answer': answer,
                        'label': label
                    }
                )

        return vector_cache
