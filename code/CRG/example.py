'''
Example usage of CRG API

Author: Ian Jackson
Date: 03/18/2025
'''

#== Imports ==#
import os
import time
import pyttsx3
import psutil
import threading

from contextlib import contextmanager
from crg_api import CRG
from crg_api import ClassifyMethod, RetrieveMethod, ExtractMethod

#== Global Variables ==#
test_dataset = {'data': [
    {'question': 'INIT', 'label': 'Degree Programs'},
    {'question': 'INIT', 'label': 'Degree Programs'},
    {'question': 'INIT', 'label': 'Degree Programs'},
    {'question': 'INIT', 'label': 'Degree Programs'},
    {'question': 'INIT', 'label': 'Degree Programs'},
    {'question': 'What degree programs does the department offer?', 'label': 'Degree Programs'},
    {'question': 'What dual degrees can I pursue?', 'label': 'Degree Programs'},
    {'question': 'What are the various research areas in the Lane Department?', 'label': 'Research Opportunities'},
    {'question': 'What research is done in the biometrics field?', 'label': 'Research Opportunities'},
    {'question': 'What are the student orgs I can join as a LCSEE student?', 'label': 'Clubs and Organizations'},
    {'question': 'What kind of activities do CyberWVU students do?', 'label': 'Clubs and Organizations'},
    {'question': 'What can I do with a computer engineering degree?', 'label': 'Career Opportunities'},
    {'question': 'What can I do with a computer science degree?', 'label': 'Career Opportunities'},
    {'question': 'What type of internships do students get?', 'label': 'Internships'},
    {'question': 'How can students get internships?', 'label': 'Internships'},
    {'question': 'What type of scholarships are available for incoming students?', 'label': 'Financial Aid and Scholarships'},
    {'question': 'How can freshmen get scholarships?', 'label': 'Financial Aid and Scholarships'},
    {'question': 'How can I get into contact with the Lane Department?', 'label': 'Location and Contact'},
    {'question': 'Where is the Lane Department Located?', 'label': 'Location and Contact'},
]}

INIT_QU = 5

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

#== Main Execution ==#
def main():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    # set dataset path
    dataset_pth = '../dataset.json'

    # change the model parameters
    classify_method = ClassifyMethod.LR
    extract_method = ExtractMethod.NER
    retrieve_method = RetrieveMethod.EKI

    # print the model parameters
    print(f'[i] Running CRG with the following parameters: {classify_method.name}, {extract_method.name}, {retrieve_method.name}')

    # init CRG
    crg = CRG(
        dataset_pth, 
        classify_method=classify_method, 
        extract_method=extract_method,
        retrieve_method=retrieve_method,
        print_info=False)
    
    # test dataset
    avg_time = 0
    avg_cpu_time = 0
    avg_cpu_usage = 0
    avg_ram_usage = 0

    print('[i] Starting to answer questions from the test dataset...\n')

    for i,question in enumerate(test_dataset['data']):
        user_question = question['question']
        
        with cpu_usage_monitor() as metrics:
            answer = crg.answer_question(user_question)

        if user_question != 'INIT':
            avg_time += metrics['wall_time']
            avg_cpu_time += metrics['cpu_time']
            avg_cpu_usage += metrics['cpu_utilization_psutil']
            avg_ram_usage += metrics['ram_usage_avg_mb']

            print(f'Question: {user_question}\nAnswer: {answer}')
            print(f'Stats:')
            print(f'  Response time: {(metrics["wall_time"]):.2f} seconds')
            print(f'  CPU time taken: {(metrics["cpu_time"]):.2f} seconds')
            print(f'  CPU usage: {(metrics["cpu_utilization_psutil"]):.2f}%')
            print(f'  Avg RAM usage: {(metrics["ram_usage_avg_mb"]):.2f} MB\n')


    n = len(test_dataset['data']) - INIT_QU  # Exclude the first 5 INIT questions
    avg_time /= n
    avg_cpu_time  /= n
    avg_cpu_usage /= n
    avg_ram_usage /= n

    print('Overall Statistics:')
    print(f' Total questions answered: {len(test_dataset["data"]) - INIT_QU}')
    print(f' Average time taken for answering questions: {avg_time:.3f} seconds')
    print(f' Average CPU time taken for answering questions: {avg_cpu_time:.3f} seconds')
    print(f' Average CPU usage: {avg_cpu_usage:.2f}%')
    print(f' Average RAM usage: {avg_ram_usage:.2f} MB\n')

    # while loop to ask questions
    # user_ans = ''
    # while user_ans != 'exit':
    #     user_ans = input('Type your question (or type "exit" to quit): ')
    #     if user_ans.lower() == 'exit':
    #         break
    #     st = time.time()
    #     answer = crg.answer_question(user_ans)
    #     et = time.time()
    #     print(f'Answer: {answer}')
    #     print(f'Time taken: {et - st:.2f} seconds\n')

    #     engine.say(answer)
    #     engine.runAndWait()

    
    # question = 'What degrees are offered for undergraduate?'

    # st = time.time()
    # answer = crg.answer_question(question)
    # et = time.time()

    # print(f'Question: {question}\nAnswer: {answer}')
    # print(f'Time taken: {et - st:.2f} seconds')

if __name__ == '__main__':
    main()