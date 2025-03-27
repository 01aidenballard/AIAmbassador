'''
Conversate
The User Interface between our user and the CRG API

Author: Aiden Ballard
Date: 03/26/2025
'''

import re
import pyttsx3 as tts
import sys
import os

# Add the CRG directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRG')))

# Now you can import the modules
from crg_api import CRG, ClassifyMethod, RetrieveMethod, ExtractMethod


def main():
    # set dataset path
    dataset_pth = '../dataset.json'

    # change the model parameters
    classify_method = ClassifyMethod.LR
    extract_method = ExtractMethod.VEC
    retrieve_method = RetrieveMethod.CSS_VEC

    # init CRG
    crg = CRG(
        dataset_pth, 
        classify_method=classify_method, 
        extract_method=extract_method,
        retrieve_method=retrieve_method,
        print_info=False)

    # answer a sample question
    question = 'What degrees are offered for undergraduate?'
    answer = crg.answer_question(question)

    print(f'Question: {question}\nAnswer: {answer}')

if __name__ == '__main__':
    main()
