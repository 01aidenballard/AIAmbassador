'''
Example usage of CRG API

Author: Ian Jackson
Date: 03/18/2025
'''

#== Imports ==#
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
        print_info=True)

    # answer a sample question
    question = 'What degrees are offered for undergraduates?'
    crg.answer_question(question)

if __name__ == '__main__':
    main()