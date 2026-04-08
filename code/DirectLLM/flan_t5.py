import json
import argparse
import time
import os
import psutil
import threading
from contextlib import contextmanager
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict
import evaluate
import sacrebleu

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

# MODELS 
# - FLAN-T5 (google/flan-t5-small) [flan_t5_finetuned]
# - ByT5 (google/byt5-small) [by_t5_finetuned]

# - MiniLM (microsoft/MiniLM-L12-H384-uncased) [MiniLM_L12_H384_finetuned]
# - DistilBERT (distilbert-base-uncased)

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

# Load the dataset.json
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
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

# Preprocess the dataset into SQuAD-like format
def preprocess_dataset(data):
    records = []
    for item in data["data"]:
        title = item["title"]
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            # Example: Populate with synthetic questions and answers
            qas = paragraph.get("qas", [])
            for qa in qas:
                question = qa["question"]
                answer = qa["answer"][0] if qa["answer"] else "Answer not provided"
                records.append({"context": context, "question": question, "answer": answer})
    return records

# Create a HuggingFace Dataset
def create_hf_dataset(records):
    contexts, questions, answers = zip(*[(r["context"], r["question"], r["answer"]) for r in records])
    dataset = Dataset.from_dict({"context": contexts, "question": questions, "answer": answers})
    return DatasetDict({"train": dataset})

# Fine-tune FLAN-T5
def fine_tune_model(dataset, model_name="google/flan-t5-small", is_hpc=False):
    if is_hpc:
        model_name = "/scratch/agb00033/AIAmbassador/code/DirectLLM/flan-t5-small"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the dataset
    def preprocess_function(examples):
        inputs = [f"question: {q} \\n context: {c}" for q, c in zip(examples["question"], examples["context"])]
        targets = examples["answer"]

        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=128, 
                truncation=True, 
                padding="max_length"
            ).input_ids

        labels = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels
        ]

        model_inputs["labels"] = labels

        return model_inputs
    
    def clean_dataset(dataset):
        # Remove unnecessary fields
        dataset.pop("context", None)
        dataset.pop("question", None)
        dataset.pop("answer", None)
        return dataset

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(clean_dataset)

    print(f"{bcolors.OKBLUE}[i] Dataset Size: {len(tokenized_dataset['train'])}{bcolors.ENDC}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./flan_t5_finetuned",
        eval_strategy="epoch", 
        weight_decay=0.01, 
        logging_steps=5,  
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        save_total_limit=2,
        fp16=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()
    model.save_pretrained("./flan_t5_finetuned")
    tokenizer.save_pretrained("./flan_t5_finetuned")

# Load the fine-tuned model and tokenizer
def load_fine_tuned_model(model_dir="./flan_t5_finetuned"):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return model, tokenizer

# Generate an answer using the fine-tuned model
def generate_answer(model, tokenizer, question, context=""):
    # Prepare the input text in the correct format
    input_text = f"question: {question} \\n context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Generate the answer
    outputs = model.generate(inputs.input_ids, max_length=100, num_beams=2, early_stopping=True, repetition_penalty=2.0)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Define F1 Score Calculation
def calculate_f1(predictions, references):
    """
    Calculate F1 score for the model's predictions against reference answers.
    """
    metric = evaluate.load("squad")  # Use SQuAD metric for F1
    results = metric.compute(predictions=predictions, references=references)
    return results["f1"]

def calculate_bleu(predictions, references):
    """
    Calculate BLEU score for the model's predictions against reference answers.
    """
    # BLEU expects lists of sentences
    reference_texts = [[ref["answers"][0]["text"]] for ref in references]
    prediction_texts = [pred["prediction_text"] for pred in predictions]
    
    bleu_score = sacrebleu.corpus_bleu(prediction_texts, reference_texts)
    return bleu_score.score

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Force retrain the model")
    parser.add_argument('--hpc', action='store_true', help="Running on HPC (only valid for my env lol)")
    args = parser.parse_args()

    if args.train:
        dataset_path = "../dataset.json"  # Path to your dataset
        raw_data = load_dataset(dataset_path)
        records = preprocess_dataset(raw_data)
        hf_dataset = create_hf_dataset(records)
        fine_tune_model(hf_dataset, is_hpc=args.hpc)

    model_dir = "./flan_t5_finetuned"  # Path to your fine-tuned model directory
    model, tokenizer = load_fine_tuned_model(model_dir)

    # Test questions and context
    test_data = load_testset("../test_dataset.json")  # Path to your test dataset
    test_data = test_data['data']
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
    # cahce of QA to evaluate
    '''
        {
            "question": "", 
            "context": "",
            "answer" : ""
        }

        {
            "question": "What are the research areas?", 
            "context": "Provide an overview of research areas and opportunities.",
            "answer" : "The department specializes in AI, robotics, cybersecurity, and computational theory."
        }
    '''

    print(f"{bcolors.OKBLUE}FLAN-T5:{bcolors.ENDC}")
    predictions = []
    references = []

    total_time = 0
    total_cpu_time = 0
    total_cpu_utilization = 0
    tot_avg_ram_usage = 0
    tot_correct = 0
    num_questions = 0

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
