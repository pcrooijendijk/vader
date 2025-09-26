import os
import re
import json
from datasets import Dataset  
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import torch

# Loading the tokenizer and PEFT model
tokenizer = AutoTokenizer.from_pretrained("./results_fine_tuning/checkpoint-42")

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    device_map="auto",         # ensures real weights are loaded
    load_in_4bit=True          # or load_in_8bit=True if that’s what you used
)

model = PeftModel.from_pretrained(model, "./results_fine_tuning/checkpoint-42")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

vader_dataset = "vader_languages.csv"
df = pd.read_csv(vader_dataset)

# Removing the following columns to only get the ID, CWE, severity score, explanation and programming language.
df = df.drop(columns=['Unnamed: 0', 'Repository', 'Submitted At', 'Approved At', 'num_files', 'num_languages'])

# Only get the samples which have a suggested fix in them
df = df[df['Description'].str.contains("Suggested Fix", case=False)]

# Path to the patches/diffs
diff_path = "./cases/"

# Add diff column initialized as empty
df["diff"] = None

# Loop through .patch files and add to df
for root, _, files in os.walk(diff_path):
    for filename in files:
        if filename.endswith(".patch"):
            filepath = os.path.join(root, filename)

            # Extract the first number from filename (case ID or index)
            match = re.search(r'\d+', filename)
            if not match:
                continue
            case_id = int(match.group())

            # Find matching row (assuming your ID column contains these numbers)
            row_match = df[df["Case"] == case_id]
            if not row_match.empty:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        diff = f.read()
                    df.loc[df["Case"] == case_id, "diff"] = diff
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

# Build dataset with prompt + explanation
def build_prompt(row):
    return f"""Patch:
{row['diff']}

Metadata:
CWE: {row['CWE']}
Severity: {row['Severity']}
Language: {row['language']}

### Explanation:
"""

dataset = Dataset.from_pandas(df)

dataset = dataset.map(lambda row: {
    "prompt": build_prompt(row),
    "label": row["Description"]
})

# Removing this redundant column __index_level_0__
dataset = dataset.remove_columns("__index_level_0__")

MAX_LENGTH = 1024 
# Function for tokenizing the input
def tokenizing(sample):
    prompt = sample["prompt"]
    label = sample["label"]
    
    full_text = prompt + label

    # Tokenize full text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    # Tokenize prompt only (to mask)
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )["input_ids"]

    labels = [-100] * len(prompt_ids) + tokenized["input_ids"][len(prompt_ids):]
    labels = labels[:MAX_LENGTH] + [-100] * (MAX_LENGTH - len(labels))

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Split 80/20 for training and validation
dataset = dataset.train_test_split(test_size=0.2, seed=42)
prompt_dict = {}

try:
    for index, datasample in enumerate(dataset["test"]):
        inputs = tokenizer(datasample["prompt"], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        prompt_dict[index] = {
            "prompt": datasample["prompt"],
            "expected_answer": datasample["label"],
            "generated_answer": tokenizer.decode(outputs[0], skip_special_tokens=True),
        }
except torch.OutOfMemoryError as e: 
    print(e)
    pass

with open("results.json", "w") as f: 
    json.dump(prompt_dict, f, indent=4)