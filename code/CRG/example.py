'''
Example usage of CRG API

Author: Ian Jackson
Date: 03/18/2025
'''

#== Imports ==#
import time

from crg_api import CRG
from crg_api import ClassifyMethod, RetrieveMethod, ExtractMethod

#== Main Execution ==#
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

    st = time.time()
    answer = crg.answer_question(question)
    et = time.time()

    print(f'Question: {question}\nAnswer: {answer}')
    print(f'Time taken: {et - st:.2f} seconds')

if __name__ == '__main__':
    main()