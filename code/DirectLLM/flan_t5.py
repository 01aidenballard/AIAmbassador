import json
import argparse
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict
import evaluate
import sacrebleu

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

# Load the dataset.json
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

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
        model_name = "/scratch/isj0001/models/flan-t5-small-local/"

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
        fine_tune_model(hf_dataset)

    model_dir = "./flan_t5_finetuned"  # Path to your fine-tuned model directory
    model, tokenizer = load_fine_tuned_model(model_dir)

    # Test questions and context
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

    # cahce of QA to evauluate
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
    num_questions = 0

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
