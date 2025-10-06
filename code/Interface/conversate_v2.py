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


class Conversation():
    """
    Class to handle the conversation between the user and the CRG API.
    """
    def __init__(self, dataset_path: str, classify_method: ClassifyMethod, extract_method: ExtractMethod, retrieve_method: RetrieveMethod, generate_method: GenerateMethod, wake_word: str = "lane", sleep_word: str = "go to sleep"):
        """
        Initialize the Conversation class with the CRG API and speech recognizer.
        """
        Log.log("SYSTEM", "Initializing conversation...")

        self.crg = CRG(
            dataset_path, 
            classify_method=classify_method, 
            extract_method=extract_method,
            retrieve_method=retrieve_method,
            generate_method=generate_method,
            print_info=False)
        
        self.lain = L(wake_word, sleep_word)

        self.listening_responses=["Yes?", "How can I help you?", "I'm listening", "What can I do for you?", "How can I assist you?", "What would you like to know?", "What is question?", "How can I be of service?"]

        self.continuation_responses=["Do you have any other questions?", "Is there anything else you'd like to know?", "Can I help you with something else?", "Would you like to ask another question?", "Is there more you'd like to discuss?", "Do you need further assistance?", "Are there additional topics you're curious about?"]

        self.previous_answer = ""


    def respond(self, question: str): 

        """
        Process the user audio and return the answer from CRG API
        
        Inputs:
            user_audio: str - the user's question in text form

        Returns:
            answer_text: str - the answer from CRG API, with a follow-up question if applicable
        """
        
        # process response
        st = time.time()
        answer = self.crg.answer_question(question)
        et = time.time()
        
        answer_text = answer['generated_answer']
        answer_class = answer['question_class']


        if answer_class != "Follow Up" and answer_class != "Repeat":

            random.seed(time.time())
            followup = self.continuation_responses[random.randint(0, len(self.continuation_responses)-1)]
            et = time.time()
            Log.log("INFO", f"Answer: {answer_text}\nFollow Up: {followup}\n(Time taken: {et - st:.2f} seconds)")

            answer = f"{answer_text} {followup}"
            self.previous_answer = answer_text
            
        
        elif answer_class == "Repeat":

            et = time.time()
            Log.log("INFO", f"Answer: {answer_text} {self.previous_answer}\n(Time taken: {et - st:.2f} seconds)")

            answer = f"{answer_text} {self.previous_answer}"
        
        else:
            Log.log("INFO", f"Answer: {answer_text}\n(Time taken: {et - st:.2f} seconds)")
            self.previous_answer = answer_text


        print(f"Previous Answer set to: {self.previous_answer}")

        return answer
    
    def conversate(self, conversation):
        """
        Function to handle the conversation between the user and the CRG API.
        """

        # while loop to ask questions
        while True:


            Log.log("SYSTEM", "Waiting for user...")
            # input type to type, speak to speak
            interaction = input()
            if interaction == "type":
                
                while True:
                    user_statement = input("Enter your question: ")

                    Log.log("INFO", f"User question: {user_statement}")

                    conversation.respond(user_statement)



            
            elif interaction == "speak":
                action = L.listen_for_action_word(self.lain)
            
                if action:
                    
                    Log.log("SYSTEM", "Wake word detected, listening for user question...")
                    # Use a random response from the listening_responses
                    random.seed(time.time())
                    response = self.listening_responses[random.randint(0, len(self.listening_responses)-1)]
                    sys_command(f"flite -voice rms -t '{response}'")
                    Log.log("INFO", f"Lain: {response}")

                    null_count = 0

                    while True:
                        
                        # listen for user question
                        user_statement = L.listen(self.lain)
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

                        answer = conversation.respond(user_statement) # process response
                        
                        

                        # print(f'Answer: {answer}')
                        # print(f'Time taken: {et - st:.2f} seconds\n')
                        

                        # completed_response = subprocess.run(f"flite -voice rms -t \"{answer}", shell=True, check=True)
                        continue # go back to listening for another question

                elif not action:
                    Log.log("SYSTEM", "Sleep word detected, exiting...")
                    sys_command("flite -voice rms -t 'Goodbye'")
                    break
        


def main():

    conversation = Conversation(
        dataset_path = '../dataset.json',
        classify_method = ClassifyMethod.SVM,
        extract_method = ExtractMethod.VEC,
        retrieve_method = RetrieveMethod.CSS_VEC,
        generate_method = GenerateMethod.CONTEXT_ONLY,
        wake_word="lane",
        sleep_word="go to sleep"
    )

    conversation.conversate(conversation)
    
        

def sys_command(command):
    os.system(command)

if __name__ == '__main__':
    main()
