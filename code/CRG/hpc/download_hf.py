from transformers import AutoTokenizer

model_name = "bert-base-uncased"
save_path = "/scratch/isj0001/huggingface_models/bert-base-uncased"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)