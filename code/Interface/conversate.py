'''
Conversate
The User Interface between our user and the CRG API

Author: Aiden Ballard
Date: 03/26/2025
'''

import time
import pyttsx3
import sys
import os
import speech_recognition_api as sr


# Add the CRG directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRG')))

# Now you can import the modules
from crg_api import CRG, ClassifyMethod, RetrieveMethod, ExtractMethod


def main():
    
    engine = pyttsx3.init(driverName = 'espeak')
    engine.setProperty('rate', 150)


    # set dataset path
    dataset_pth = '../dataset.json'

    # change the model parameters
    classify_method = ClassifyMethod.SVM
    extract_method = ExtractMethod.NER
    retrieve_method = RetrieveMethod.EKI

    # init CRG
    crg = CRG(
        dataset_pth, 
        classify_method=classify_method, 
        extract_method=extract_method,
        retrieve_method=retrieve_method,
        print_info=False)

    # while loop to ask questions
    while True:
        print("Please type 'q' when you are ready to ask a question, or 'exit' to quit: ")
        user_ans = input()
        if user_ans == 'q':
           
    
            engine.stop() # free resources for mic
            print("One moment!")
            user_q = sr.speech_recognition()
             
            if user_q is None:
                print("Could not understand audio or no speech detected.")
                continue
            st = time.time()
            answer = crg.answer_question(user_q)
            et = time.time()
            print(f'Answer: {answer}')
            print(f'Time taken: {et - st:.2f} seconds\n')

            engine.say(answer)
            engine.runAndWait()
            engine.stop()
        elif user_ans == 'exit':
            break

    
    # question = 'What degrees are offered for undergraduate?'

    # st = time.time()
    # answer = crg.answer_question(question)
    # et = time.time()

    # print(f'Question: {question}\nAnswer: {answer}')
    # print(f'Time taken: {et - st:.2f} seconds')


if __name__ == '__main__':
    main()
