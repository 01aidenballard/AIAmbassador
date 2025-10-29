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
import argparse
import json
import psutil
import threading

from contextlib import contextmanager

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
    def __init__(self, method_rag: bool, dataset_path: str, classify_method: ClassifyMethod, extract_method: ExtractMethod, retrieve_method: RetrieveMethod, generate_method: GenerateMethod, wake_word: str = "lane", sleep_word: str = "go to sleep"):
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
            method_rag=method_rag,
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

        print(f"Answer Document: {answer_class}")


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


        #print(f"Previous Answer set to: {self.previous_answer}")

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
                        

                        completed_response = subprocess.run(f"flite -voice rms -t \"{answer}", shell=True, check=True)
                        continue # go back to listening for another question

                elif not action:
                    Log.log("SYSTEM", "Sleep word detected, exiting...")
                    sys_command("flite -voice rms -t 'Goodbye'")
                    break

#== Method ==#
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
        dbanswer = ql['answer']
        question = ql['question']

        labeled_data['data'].append(
            {
                'question': question,
                'answer': dbanswer
            }
        )

    return labeled_data


@contextmanager
def cpu_usage_monitor(sample_interval=0.05):
    process = psutil.Process(os.getpid())
    memory_samples = []
    running = True

    # mem based sampling function
    def sample_memory():
        while running:
            memory_samples.append(process.memory_info().rss)
            time.sleep(sample_interval)

    # start memory sampling
    sampler_thread = threading.Thread(target=sample_memory)
    sampler_thread.start()

    # before execution
    start_wall = time.time()
    start_cpu = process.cpu_times().user + process.cpu_times().system
    start_mem = process.memory_info().rss
    process.cpu_percent(interval=None)

    try:
        # during execution
        yield_value = {}
        yield yield_value
    finally:
        # after execution
        end_wall = time.time()
        end_cpu = process.cpu_times().user + process.cpu_times().system
        end_mem = process.memory_info().rss
        end_cpu_percent = process.cpu_percent(interval=None)

        # stop memory sampling
        running = False
        sampler_thread.join()

        # calculate averages
        if memory_samples:
            avg_ram_usage_bytes = sum(memory_samples) / len(memory_samples)
        else:
            avg_ram_usage_bytes = 0

        # calc metrics
        wall_time_elapsed = end_wall - start_wall
        cpu_time_used = end_cpu - start_cpu
        cpu_utilization_percent = (cpu_time_used / wall_time_elapsed) * 100
        ram_used_bytes = end_mem - start_mem
        ram_used_mb = ram_used_bytes / (1024 * 1024)

        # store in dict
        yield_value.update({
            'wall_time': wall_time_elapsed,
            'cpu_time': cpu_time_used,
            'cpu_utilization_calculated': cpu_utilization_percent,
            'cpu_utilization_psutil': end_cpu_percent,
            'ram_usage_change_mb': ram_used_mb,
            'ram_usage_start_mb': start_mem / (1024 * 1024),
            'ram_usage_end_mb': end_mem / (1024 * 1024),
            'ram_usage_avg_mb': avg_ram_usage_bytes / (1024 * 1024)  # Convert to MB
        })

def main(args):

    conversation = Conversation(
        dataset_path = '../dataset.json',
        classify_method = ClassifyMethod.SVM,
        extract_method = ExtractMethod.VEC,
        retrieve_method = RetrieveMethod.CSS_VEC,
        generate_method = GenerateMethod.FLAN_T5,
        method_rag = True,
        wake_word="lane",
        sleep_word="go to sleep"
    )

    if args.test:
        print("Running on test dataset...")
        test_dataset = load_testset(os.path.join("..", "test_dataset.json"))

        avg_time, avg_cpu_time, avg_cpu_usage, avg_ram_usage, avg_distance = 0, 0, 0, 0, 0

        for QA in test_dataset['data']:
            question = QA['question']
            dbanswer = QA['answer']
            
            
            with cpu_usage_monitor() as metrics:
                answer = conversation.respond(question)

            avg_time += metrics['wall_time']
            avg_cpu_time += metrics['cpu_time']
            avg_cpu_usage += metrics['cpu_utilization_psutil']
            avg_ram_usage += metrics['ram_usage_avg_mb']

            print(f'Question: {question}\nAnswer: {answer}')
            print(f'Stats:')
            print(f'  Response time: {(metrics["wall_time"]):.2f} seconds')
            print(f'  CPU time taken: {(metrics["cpu_time"]):.2f} seconds')
            print(f'  CPU usage: {(metrics["cpu_utilization_psutil"]):.2f}%')
            print(f'  Avg RAM usage: {(metrics["ram_usage_avg_mb"]):.2f} MB\n')

        n = len(test_dataset['data'])
        avg_time /= n
        avg_cpu_time  /= n
        avg_cpu_usage /= n
        avg_ram_usage /= n
        avg_distance /= n

        print('Overall Statistics:')
        print(f' Total questions answered: {n}')
        print(f' Average time taken for answering questions: {avg_time:.3f} seconds')
        print(f' Average CPU time taken for answering questions: {avg_cpu_time:.3f} seconds')
        print(f' Average CPU usage: {avg_cpu_usage:.2f}%')
        print(f' Average RAM usage: {avg_ram_usage:.2f} MB')

    else:
        conversation.conversate(conversation)
    
        

def sys_command(command):
    os.system(command)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='User Conversation Interface')

    argparser.add_argument('--test', action='store_true', help='Run on test dataset')

    args = argparser.parse_args()

    main(args)
