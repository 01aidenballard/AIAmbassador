'''
Example usage of CRG API

Author: Ian Jackson
Date: 03/18/2025
'''

#== Imports ==#
import time
import pyttsx3

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

#== Main Execution ==#
def main():
    engine = pyttsx3.init()
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
    
    # test dataset
    avg_time = 0
    for i,question in enumerate(test_dataset['data']):
        user_question = question['question']

        st = time.time()
        answer = crg.answer_question(user_question)
        et = time.time()

        if i>5: avg_time += (et - st)

        print(f'Question: {user_question}\nAnswer: {answer}')
        print(f'Time taken: {et - st:.2f} seconds\n')

    avg_time /= (len(test_dataset['data']) - 5)  # Exclude the first 3 INIT questions
    print(f'Average time taken for answering questions: {avg_time:.3f} seconds\n')

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