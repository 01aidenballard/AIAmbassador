'''
Conversate
The User Interface between our user and the CRG API

Author: Aiden Ballard
Date: 03/26/2025
'''

import time
import sys
import os
import random
from speech_recognition_api import Listen as L

# Add the CRG directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRG')))

from crg_api import CRG, ClassifyMethod, ExtractMethod, RetrieveMethod, GenerateMethod

# Add the Log directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Logs')))

from Logging import Log

#== Global Variables ==#

listening_responses=["Yes?", "How can I help you?", "I'm listening", "What can I do for you?", "How can I assist you?", "What would you like to know?", "What is question?", "How can I be of service?"]


def main():
    
    # set dataset path
    dataset_pth = '../dataset.json'

    # change the model parameters
    classify_method = ClassifyMethod.SVM
    extract_method = ExtractMethod.VEC
    retrieve_method = RetrieveMethod.CSS_VEC
    generate_method = GenerateMethod.CONTEXT_ONLY

    # init CRG
    Log.log("SYSTEM", "Initializing conversation...")
    crg = CRG(
        dataset_pth, 
        classify_method=classify_method, 
        extract_method=extract_method,
        retrieve_method=retrieve_method,
        generate_method=generate_method,
        print_info=False)
    
    # init Speech Recognition
    lain = L(device_name="Microphone (Realtek High Definition Audio)",)

    # while loop to ask questions
    while True:
        Log.log("SYSTEM", "Waiting for user...")
        action = L.listen_for_action_word(lain)
        if action:
           
            Log.log("SYSTEM", "Wake word detected, listening for user question...")
            # Use a random response from the listening_responses
            random.seed(time.time())
            response = listening_responses[random.randint(0, len(listening_responses)-1)]
            sys_command(f"flite -voice rms -t '{response}'")
            Log.log("INFO", f"Lain: {response}")

            user_q = L.listen(lain)
            Log.log("INFO", f"User question: {user_q}")
             
            if user_q is None:
                # error = "Error: Could not understand question"
                # sys_command(error)
                Log.log("INFO", "No question detected, sleeping...")
                continue

            st = time.time()
            answer = crg.answer_question(user_q)
            et = time.time()
            Log.log("INFO", f"Answer: {answer}\n(Time taken: {et - st:.2f} seconds)")

            sys_command(f"flite -voice rms -t '{answer}'")
        elif not action:
            Log.log("SYSTEM", "Sleep word detected, exiting...")
            sys_command("flite -voice rms -t 'Goodbye'")
            break
        
            
            

def sys_command(command):
    os.system(command)

if __name__ == '__main__':
    main()
