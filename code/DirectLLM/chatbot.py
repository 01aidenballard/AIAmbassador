'''
Test env for chatbot
Author: Ian Jackson

'''

#== Imports ==#
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, pipeline, DefaultDataCollator
from datasets import load_dataset

from flan_t5 import calculate_bleu, calculate_f1

#== Methods ==#
def init():
    model_name = "deepset/roberta-base-squad2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # dataset = load_dataset("json", data_files={"train": "dataset.json", "validation": "dataset.json"})
    dataset = load_dataset("json", data_files="dataset.json")

    return model, tokenizer, dataset

def train(dataset, model, tokenizer):
    # Tokenize the dataset
    def tokenize_function(examples):
        # print("Example keys:", examples.keys())
        # print("Example structure:", examples['data'][0])

        contexts = []
        questions = []
        answers = []

        # examples['data'] is a list, so iterate over it directly
        for data_items in examples['data']:  # Process all items in the top-level list
            for data_item in data_items:     # Process each inner dictionary
                for paragraph in data_item['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        answer = qa['answer'][0]  # Assuming one answer per question
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
        
        tokenized = tokenizer(
            questions,
            contexts,
            truncation=True,
            padding="max_length",
            max_length=512,  # Adjust based on model and use case
            return_tensors="pt"
        )

        # Add the start and end token indices for answers
        start_positions = []
        end_positions = []

        for i, answer in enumerate(answers):
            start_idx = tokenized.char_to_token(i, 0)
            end_idx = tokenized.char_to_token(i, 0 + len(answer[0]) - 1)

            # If the answer cannot be found in the context, use 0 as the start/end positions
            if start_idx is None:
                start_idx = 0
            if end_idx is None:
                end_idx = 0

            start_positions.append(start_idx)
            end_positions.append(end_idx)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions

        return tokenized

    train_dataset = dataset['train']
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="./roberta_finetuned",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        save_total_limit=2
    )

    # training_args = TrainingArguments(
    #     output_dir="./roberta_finetuned",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    #     save_steps=500,
    #     save_total_limit=2
    # )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Pass tokenized dataset directly
        eval_dataset=tokenized_dataset,  # For simplicity, use the same dataset for evaluation
        tokenizer=tokenizer,
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)

    model.save_pretrained("./roberta_finetuned")
    tokenizer.save_pretrained("./roberta_finetuned")


def get_answer(question, context, model, tokenizer):
    """
    Get an answer to a question based on a given context using the QA model.
    """
    inputs = tokenizer.encode_plus(question,
                                   context,
                                   max_length=512,
                                   truncation=True,
                                   return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Get the most likely beginning and end of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    if answer_start < answer_end:
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )
        return answer.strip()
    else:
        return "I'm not sure about the answer to that question."

def chatbot(isTrain):
    """
    A chatbot interface for question answering.
    """
    print("Initializing...")
    model, tokenizer, dataset = init()

    if isTrain:
        train(dataset, model, tokenizer)

    print("Welcome to the Q&A chatbot! Type 'exit' to quit.")
    # context = input("Please provide the context for your questions:\n")
    # print("\nYou can now ask questions based on the provided context.\n")

    # context = """
    #     The Computer Science and Electrical Engineering (CSEE) department combines cutting-edge research with rigorous academic programs. We offer undergraduate and graduate degrees, including majors in electrical engineering, computer engineering, computer science, cybersecurity, and robotics engineering.

    #     Research Areas:
    #     - Machine Learning and Artificial Intelligence, Robotics and Embedded Systems, Biometrics

    #     Computer Science Career Paths:
    #     Graduates can work in software development, data science, embedded systems, academia, IT consulting, and more.

    #     Electrical Engineering Career Paths:
    #     careers in power systems, embedded systems, telecommunications, renewable energy, and more

    #     Facilities and Clubs:
    #     - AIWVU, IEEE, University Rover Challenge
    #     """

    # context = "The CSEE department offers undergraduate and graduate programs in computer science and electrical engineering."

    qa_pipeline = pipeline("question-answering", model="./roberta_finetuned", tokenizer=tokenizer)

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
        }
        
    ]

    predictions = []
    references = []

    for item in test_data:
        question = item["question"]
        context = item["context"]
        ground_truth = item["answer"]

        print(f"\nQuestion: {question}")
        answer = get_answer(question, context, model, tokenizer)
        # result = qa_pipeline(question=question, context=context)
        # answer = result['answer']
        print(f"Answer: {answer}")

        # Store predictions and references for metrics
        predictions.append({"id": str(len(predictions) + 1), "prediction_text": answer})
        references.append({"id": str(len(references) + 1), "answers": [{"text": ground_truth, "answer_start": 0}]})

    f1_score = calculate_f1(predictions, references)
    bleu_score = calculate_bleu(predictions, references)

    print(f"\nF1 Score: {f1_score}")
    print(f"BLEU Score: {bleu_score}")

    # while True:
    #     question = input("Your question: ")
    #     if question.lower() in ["exit", "quit"]:
    #         print("Goodbye!")
    #         break
        
    #     # answer = get_answer(question, context, model, tokenizer)
    #     result = qa_pipeline(question=question, context=context)
    #     answer = result['answer']
    #     if answer:
    #         print(f"Answer: {answer}")
    #     else:
    #         print("I'm not sure about the answer to that question.")

# Run the chatbot
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Force retrain the model")
    args = parser.parse_args()
    chatbot(args.train)