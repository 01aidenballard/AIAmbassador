import json
import argparse
import torch
import time
import os
import psutil
import threading
from contextlib import contextmanager
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from flan_t5 import calculate_bleu, calculate_f1

EVAL_RESP = True

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
        label = ql['label']
        question = ql['question']
        context = ql['context']
        answer = ql['answer']

        labeled_data['data'].append(
            {
                'question': question,
                'label': label,
                'context': context,
                'answer': answer
            }
        )

    return labeled_data


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

    test_dataset_path = '../test_dataset.json'
    test_data = load_testset(test_dataset_path)
    test_data = test_data['data']

    predictions = []
    references = []

    total_time = 0
    total_cpu_time = 0
    total_cpu_utilization = 0
    tot_avg_ram_usage = 0
    tot_correct = 0
    num_questions = 0

    print(f"{bcolors.OKBLUE}BART:{bcolors.ENDC}")

    for item in test_data:
        question = item["question"]
        context = item["context"]
        ground_truth = item["answer"]

        print(f"\nQuestion: {question}")
        with cpu_usage_monitor() as monitor:
            answer = generate_answer(model, tokenizer, question, context)
        print(f"Answer: {answer}")

        response_time = monitor['wall_time']
        cpu_time = monitor['cpu_time']
        cpu_utilization = monitor['cpu_utilization_psutil']
        avg_ram_usage = monitor['ram_usage_avg_mb']

        total_time += response_time
        total_cpu_time += cpu_time
        total_cpu_utilization += cpu_utilization
        tot_avg_ram_usage += avg_ram_usage
        num_questions += 1

        if EVAL_RESP:
            is_correct = input(f"Is the answer correct? (y/n): ").strip().lower()
            if is_correct == 'y': tot_correct += 1

        # Store predictions and references for metrics
        predictions.append({"id": str(len(predictions) + 1), "prediction_text": answer})
        references.append({"id": str(len(references) + 1), "answers": [{"text": ground_truth, "answer_start": 0}]})

    f1_score = calculate_f1(predictions, references)
    accuracy = tot_correct / num_questions
    bleu_score = calculate_bleu(predictions, references)
    average_response_time = total_time / num_questions
    average_cpu_time = total_cpu_time / num_questions
    average_cpu_utilization = total_cpu_utilization / num_questions
    average_ram_usage = tot_avg_ram_usage / num_questions

    print(f"\nF1 Score: {f1_score:.6f}")
    print(f"Accuracy: {accuracy*100:.3f}")
    print(f"BLEU Score: {bleu_score:.6f}")
    print(f"Avg Resp Time: {average_response_time:.4f}s")
    print(f"Avg CPU Time: {average_cpu_time:.4f}s")
    print(f"Avg CPU Utilization: {average_cpu_utilization:.2f}%")
    print(f"Avg RAM Usage: {average_ram_usage:.2f} MB")


if __name__ == "__main__":
    main()
