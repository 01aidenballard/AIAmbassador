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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

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

def generate_TinyLlama(question, answer):
    '''
    Generate a response using the TinyLlama model.

    Args:
        question (str): The user's question.
        answer (str): The answer retrieved from the previous step.

    Returns:
        str: The generated response.
        '''

    # load model
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    # system prompt for TinyLlama
    system_prompt = """
        You are a friendly and engaging tour guide for West Virginia University.
        Your role is to provide clear, conversational, and helpful responses based on the given information.

        - Rephrase the provided information in a natural, engaging way.
        - Do NOT mention that the information was 'given' or 'provided'.
        - Do NOT hallucinate or make up information, only use what is given.
        - Keep responses concise.
        """

    # user prompt for TinyLlama
    user_prompt = f"Answer the Users Question: {question} with this context: {answer}"

    # put together input text
    # input_text = (
    #     f"User Question: {question}\n"
    #     f"Retrieved Answer: {answer}\n"
    #     f"Generated Response:"
    # )
    

    messages = [
        {"role": "system", "content": system_prompt},  
        {"role": "user", "content": user_prompt},  
    ]

    # tokenize input
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_ids = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
    # inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    input_length = input_ids.shape[1]  # Get the length of the input tokens

    # Generate response
    with torch.no_grad():
        output_tokens = model.generate(
            # **inputs,
            input_ids=input_ids,
            max_length=512,
            repetition_penalty=1.2,  
            num_return_sequences=1,  
            eos_token_id=tokenizer.eos_token_id 
        )

    # Decode the generated response
    response = tokenizer.decode(output_tokens[0, input_length:], skip_special_tokens=True)
    
    return response


def generate_FlanT5(question, answer):
    '''
    Generate a response using the Flan-T5 model.
    
    Args:
        question (str): The user's question.
        answer (str): The answer retrieved from the previous step.
    Returns:
        str: The generated response.
    '''

    # load model
    name = 'google/flan-t5-small'
    tokenizer = T5Tokenizer.from_pretrained(name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(name)

    # system prompt for Flan-T5
    system_prompt = """
        Persona: You are "Lain," a friendly and enthusiastic university tour guide. Your audience is a group of prospective high school students and their families. Your tone should be welcoming, helpful, and engaging.

        Core Task: Your primary goal is to take a piece of factual information (the "Retreived Answer") and rephrase it into a natural, conversational response("Generated Response") to a "User Question."

        Instructions:
        1.  Natural Language: Transform the provided "Retrieved Answer" from a factual statement into a flowing, easy-to-understand sentence or two. Imagine you are speaking directly to someone on a campus tour.
        2.  Strict Information Adherence: You MUST only use the information provided in the "Retrieved Answer." Do not add any new facts, statistics, or details, even if they seem relevant. Do not hallucinate. Be concise, but thorough with the specifics.
        3.  No Meta-Commentary: Do not mention that you have been "given" or "provided" with information. The response should be seamless.
        4.  Engage with a Question: After providing the answer, always ask a relevant, open-ended follow-up question to encourage further conversation.
        5.  Structure: The final output should only be the conversational reply from Lain.

        Example of your task:

        User Question: "What's the student-to-faculty ratio?"
        Retrieved Answer: "The student-to-faculty ratio is 15 to 1."

        Generated Response:
        "That's a great question! We have a student-to-faculty ratio of 15 to 1, which means our professors get to know their students really well. Are you interested in any particular academic departments?"

        Now, use the following information to answer the user's question:
        """

    # put together input text
    input_text = (
        f"{system_prompt}"
        f"User Question: {question}\n"
        f"Retrieved Answer: {answer}\n"
        f"Generated Response:"
    )



    # tokenize input
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

    # generate response
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
    
    return response

    


#== Main Execution ==#
def main(cr_args, args):

    avg_sim_score = 0
    avg_resp_time = 0
    avg_nli_score = 0

    nli = pipeline("text-classification", model="roberta-large-mnli")
        
    for q in range(len(test_dataset['data'])):
        # get sample question
        question = test_dataset['data'][q]['question']

        # get answer from retrieve step
        answer = r.main(cr_args, question)

        # start time for response generation
        st = time.time()

        # determine the generation method to use
        if args.gen_method == 'Flan-T5':

            response = generate_FlanT5(question, answer)

        elif args.gen_method == 'TinyLlama':
            
            response = generate_TinyLlama(question, answer)
        
        resp_time = time.time() - st
        avg_resp_time += resp_time
        # print response
        print(f'Question: {question}')
        print(f'\nRetrieved response: {answer}\n\nGenerated Response: {response}\n\nResponse Time: {resp_time:.4f} s')

        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Compute embeddings for the question and answer
        question_embedding = model.encode([answer, response])
        # Compute cosine similarity
        similarity = cosine_similarity([question_embedding[0]], [question_embedding[1]])[0][0]
        avg_sim_score += similarity

        # NLI score
        nli_score = nli(f"{answer} </s> {response}")
        avg_nli_score += nli_score[0]['score']

        print(f'Cosine Similarity: {similarity:.4f}\n{"-"*50}\n')
        print(f'NLI Score: {nli_score[0]["score"]:.4f} ({nli_score[0]["label"]})\n{"-"*50}\n')

    # Calculate averages
    avg_sim_score /= len(test_dataset['data'])
    avg_resp_time /= len(test_dataset['data'])
    avg_nli_score /= len(test_dataset['data']) 

    print(f'Average Cosine Similarity Score: {avg_sim_score:.4f}')
    print(f'Average Response Time: {avg_resp_time:.4f} seconds')
    print(f'Average NLI Score: {avg_nli_score:.4f}')

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