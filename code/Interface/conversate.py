'''
Conversate
The User Interface between our user and the CRG API

Author: Aiden Ballard
Date: 03/26/2025
'''

import time
import pyttsx3 as tts
import sys
import os

# Add the CRG directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRG')))

# Now you can import the modules
from crg_api import CRG, ClassifyMethod, RetrieveMethod, ExtractMethod

def main():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

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

    # while loop to ask questions
    user_ans = ''
    while user_ans != 'exit':
        user_ans = input('Type your question (or type "exit" to quit): ')
        if user_ans.lower() == 'exit':
            break
        st = time.time()
        answer = crg.answer_question(user_ans)
        et = time.time()
        print(f'Answer: {answer}')
        print(f'Time taken: {et - st:.2f} seconds\n')

        engine.say(answer)
        engine.runAndWait()

    
    # question = 'What degrees are offered for undergraduate?'

    # st = time.time()
    # answer = crg.answer_question(question)
    # et = time.time()

    # print(f'Question: {question}\nAnswer: {answer}')
    # print(f'Time taken: {et - st:.2f} seconds')

if __name__ == '__main__':
    main()