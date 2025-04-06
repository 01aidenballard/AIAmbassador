import json
import argparse
import torch
import time
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import evaluate
import sacrebleu

from flan_t5 import calculate_bleu, calculate_f1

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_and_process_dataset(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    data = []
    for section in dataset['data']:
        for paragraph in section['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answer'][0] if qa['answer'] else ""
                data.append({"context": context, "question": question, "answer": answer})
    return data

def fine_tune_bart(dataset, model_name="facebook/bart-base", is_hpc=False):
    def preprocess_function(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples['question'], examples['context'])]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(examples['answer'], max_length=128, truncation=True, padding="max_length")
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./bart_finetuned",
        eval_strategy="epoch",
        weight_decay=0.01,
        learning_rate=3e-5,
        logging_steps=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        save_total_limit=2,
    )

    if is_hpc:
        model_name = "/scratch/isj0001/models/bart-large-local/"
        
    model = BartForConditionalGeneration.from_pretrained(model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("./bart_finetuned")
    tokenizer.save_pretrained("./bart_finetuned")
    return model, tokenizer

def load_fine_tuned_model(model_path="./bart_finetuned"):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_answer(model, tokenizer, question, context):
    inputs = tokenizer(f"question: {question} context: {context}", return_tensors="pt", max_length=512, truncation=True).to(model.device)
    outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Force retrain the model")
    parser.add_argument('--hpc', action='store_true', help="Running on HPC (only valid for my env lol)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = '../dataset.json'
    global tokenizer

    if args.train:
        print('Training model')
        raw_data = load_and_process_dataset(dataset_path)
        dataset = Dataset.from_dict({
            "context": [item['context'] for item in raw_data],
            "question": [item['question'] for item in raw_data],
            "answer": [item['answer'] for item in raw_data],
        })

        if args.hpc:
            tokenizer = BartTokenizer.from_pretrained("/scratch/isj0001/models/bart-large-local/")
        else:
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        
        model, tokenizer = fine_tune_bart(dataset, is_hpc=args.hpc)
    else:
        print('Loading finetuned model')
        model, tokenizer = load_fine_tuned_model()

    test_data = [
        {
            "question": "What degree programs does the department offer?", 
            "context": "Describe the degree programs offered by the department.",
            "answer" : "The department offers undergraduate degrees in Computer Engineering (CPE), Computer Science (CS), Electrical Engineering (EE), and Cybersecurity (CYBE)."
        },
        {
            "question": "What dual degrees can I pursue?", 
            "context": "Describe the degree programs offered by the department.",
            "answer" : "Students can pursue a dual degree in Computer Science (CS) and Computer Engineering (CPE) or in Electrical Engineering (EE) and Computer Science (CS)"
        },
        {
            "question": "What are the various research areas in the Lane Department?", 
            "context": "Provide an overview of research areas and opportunities.",
            "answer" : "Research areas include biometric systems, AI, robotics and autonomous vehicles, big data and visualization, nanotechnology/electronics, power and energy systems, radio and astronomy, software engineering, theoretical computer science, and wireless communications and sensor networks."
        },
        {
            "question": "What research is done in the biometrics field?", 
            "context": "Provide an overview of research areas and opportunities.",
            "answer" : "Research in biometrics focuses on using biological signatures like fingerprints, voice, face, and DNA for identification or authentication in applications such as criminal justice, e-commerce, and medical fields."
        },
        {
            "question": "What are the student orgs I can join as a LCSEE student?", 
            "context": "Mention student organizations and extracurricular activities.",
            "answer" : "You can get involved in groups such as the Association for Computing Machinery, CyberWVU, Eta Kappa Nu, IEEE, Student Society for the Advancement of Biometrics, Upsilon Phi Epsilon, Women in Computer Science and Electrical Engineering, and WVU Amateur Radio Club."
        },
        {
            "question": "What kind of activities do CyberWVU students do?", 
            "context": "Mention student organizations and extracurricular activities.",
            "answer" : "CyberWVU offers activities like competitions, security training sessions, speaker events, open-source volunteer projects, and tutoring hours."
        },
        {
            "question": "What can I do with a computer engineering degree?", 
            "context": "Discuss career paths for graduates.",
            "answer" : "Computer engineering graduates can pursue careers in embedded systems, robotics, hardware design, software-hardware integration, telecommunications, and IoT development."
        },
        {
            "question": "What can I do with a computer science degree?", 
            "context": "Discuss career paths for graduates.",
            "answer" : "Computer science graduates can enter professions like software development, data science, artificial intelligence, cybersecurity, game design, database management, and IT consulting."
        }
        
    ]

    predictions = []
    references = []

    total_time = 0
    num_questions = 0

    print(f"{bcolors.OKBLUE}BART:{bcolors.ENDC}")

    for item in test_data:
        question = item["question"]
        context = item["context"]
        ground_truth = item["answer"]

        print(f"\nQuestion: {question}")
        start_time = time.time()
        answer = generate_answer(model, tokenizer, question, context)
        end_time = time.time()
        print(f"Answer: {answer}")

        response_time = end_time - start_time
        total_time += response_time
        num_questions += 1

        # Store predictions and references for metrics
        predictions.append({"id": str(len(predictions) + 1), "prediction_text": answer})
        references.append({"id": str(len(references) + 1), "answers": [{"text": ground_truth, "answer_start": 0}]})

    f1_score = calculate_f1(predictions, references)
    bleu_score = calculate_bleu(predictions, references)
    average_response_time = total_time / num_questions

    print(f"\nF1 Score: {f1_score}")
    print(f"BLEU Score: {bleu_score}")
    print(f"Avg Resp Time: {average_response_time:.4f}s")

if __name__ == "__main__":
    main()
