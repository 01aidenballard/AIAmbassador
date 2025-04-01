'''
CRG - Generate Research
Traditional ML Implementation

Author: Ian Jackson
Date: 03-17-2025

'''

#== Imports ==#
import os
import sys
import json
import time
import argparse
import torch
import spacy

sys.path.insert(1, './retrieve')

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import retrieve_API as r # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

#== Global Variables ==#
test_dataset = {'data': [
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

#== Classes ==#


#== Methods ==#


#== Main Execution ==#
def main(cr_args, args):
    # get sample question
    question = test_dataset['data'][0]['question']

    # get answer from retrieve step
    answer = r.main(cr_args, question)

    # system prompt
    system_prompt = """
        You are a friendly and engaging university tour guide.
        Your role is to provide clear, conversational, and helpful responses based on the given information.

        - Rephrase the provided information in a natural, engaging way.
        - Do NOT mention that the information was 'given' or 'provided'—just answer conversationally.
        - Keep responses concise but informative.
        - After answering, ask a relevant follow-up question to continue the conversation.
        """


    # put together input text
    input_text = (
        f"User Question: {question}\n"
        f"Retrieved Answer: {answer}\n"
        f"Generated Response:"
    )

    # determine the generation method to use
    if args.gen_method == 'Flan-T5':
        # load model
        name = 'google/flan-t5-small'
        tokenizer = T5Tokenizer.from_pretrained(name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(name)

        # tokenize input
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

        # generate response
        st = time.time()
        output_tokens = model.generate(
            **inputs,
            max_length=256,
            do_sample=True,  # Enables creative responses
            temperature=0.7,  # Introduces variety
            top_p=0.9,  # Ensures diverse and high-quality generation
            repetition_penalty=1.2,  # Prevents repeating phrases
            num_return_sequences=1,  # Single response
            eos_token_id=tokenizer.eos_token_id  # Ensures proper sentence ending
        )

        # decode the generated response
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        resp_time = time.time() - st
    
        # print response
        print(f'Question: {question}')
        print(f'\nRetrieved response: {answer}\n\nGenerated Response: {response}\n\nResponse Time: {resp_time:.4f} ms')

    elif args.gen_method == 'TinyLlama':
        # load model
        model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)

        messages = [
            {"role": "system", "content": system_prompt},  
            {"role": "user", "content": question},  
            {"role": "assistant", "content": answer}  # Assistant uses this as background info
        ]

        # tokenize input
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
        # inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate response
        st = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_ids,
                max_length=256,
                repetition_penalty=1.2,  
                num_return_sequences=1,  
                eos_token_id=tokenizer.eos_token_id 
            )

        # Decode the generated response
        response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        resp_time = time.time() - st

        # print response
        print(f'Question: {question}')
        print(f'\nRetrieved response: {answer}\n\nGenerated Response: {response}\n\nResponse Time: {resp_time:.4f} ms')

if __name__ == "__main__":
    cr_args = {
        'classify_method': 'LR',
        'extract_method': 'vec',
        'retrieve_method': 'CSS-vec'
    }

    argparser = argparse.ArgumentParser(description='Generate Step')

    # arguments
    argparser.add_argument('--gen_method', type=str, choices=['Flan-T5', 'TinyLlama'], help='Generate method to use')

    args = argparser.parse_args()

    main(cr_args, args)