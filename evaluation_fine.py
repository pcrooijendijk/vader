# Loading the PEFT model and then also use it for evaluation: generating answers.

import os
import json
import argparse
from datasets import load_from_disk  
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-id', type=int)
args = parser.parse_args()

# Loading the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./results_fine_tuning/checkpoint-42")

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    device_map="auto",        
    load_in_4bit=True          
)

# Loading the PEFT model
model = PeftModel.from_pretrained(model, "./results_fine_tuning/checkpoint-42")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Loading the dataset
dataset = load_from_disk("vader2/dataset")

# Function to save and load from JSON file
def save_entry(key, value) -> None:
    results_filename = "results.json"
    data = {}
    if os.path.exists(results_filename) and os.path.getsize(results_filename):
        with open(results_filename) as f:
            data = json.load(f)
    data[key] = value
    with open(results_filename, "w") as f:
        json.dump(data, f, indent=4)

try:
    datasample_prompt = dataset["test"][args.id]["prompt"]
    datasample_label = dataset["test"][args.id]["label"]
    inputs = tokenizer(datasample_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    save_entry(args.id, {
        "prompt": datasample_prompt,
        "expected_answer": datasample_label,
        "generated_answer": tokenizer.decode(outputs[0], skip_special_tokens=True),
    })
except torch.OutOfMemoryError as e: 
    print(e)
    pass
