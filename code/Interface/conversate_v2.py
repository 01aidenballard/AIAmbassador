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
import subprocess

from speech_recognition_api import Listen as L

# Add the CRG directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CRG')))

from crg_api import CRG, ClassifyMethod, ExtractMethod, RetrieveMethod, GenerateMethod

# Add the Log directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Logs')))

from Logging import Log

#== Global Variables ==#

listening_responses=["Yes?", "How can I help you?", "I'm listening", "What can I do for you?", "How can I assist you?", "What would you like to know?", "What is question?", "How can I be of service?"]

continuation_responses=["Do you have any other questions?", "Is there anything else you'd like to know?", "Can I help you with something else?", "Would you like to ask another question?", "Is there more you'd like to discuss?", "Do you need further assistance?", "Are there additional topics you're curious about?"]

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
    lain = L(wake_word="lane", sleep_word="go to sleep")

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

            null_count = 0

            while True:
                
                # listen for user question
                user_statement = L.listen(lain)
                Log.log("INFO", f"User question: {user_statement}")
                
                if user_statement is None:
                    # error = "Error: Could not understand question"
                    # sys_command(error)
                    Log.log("ERROR", "Could not understand question, please try again...")
                    sys_command("flite -voice rms -t 'Could not understand question, please try again...'")

                    # count null messages, if 3 in a row, go back to sleep
                    null_count += 1
                    if null_count == 3:
                        Log.log("SYSTEM", "Too many null messages, going back to sleep...")
                        break
                    continue

                null_count = 0 # reset null count if we got a valid question

                st = time.time()
                answer = crg.answer_question(user_statement)
                et = time.time()

                print(f'Answer: {answer}')
                print(f'Time taken: {et - st:.2f} seconds\n')

                random.seed(time.time())
                followup = continuation_responses[random.randint(0, len(continuation_responses)-1)]
                
                Log.log("INFO", f"Answer: {answer}\nFollow Up: {followup}\n(Time taken: {et - st:.2f} seconds)")

                completed_response = subprocess.run(f"flite -voice rms -t \"{answer}. {followup}\"", shell=True, check=True)
                continue # go back to listening for another question

        elif not action:
            Log.log("SYSTEM", "Sleep word detected, exiting...")
            sys_command("flite -voice rms -t 'Goodbye'")
            break
        
            

def sys_command(command):
    os.system(command)

if __name__ == '__main__':
    main()
