# The ultimate goal is to fine tune code Llama on the VADER dataset where the input is a commit message and
# the model should output a "human" like explanation for this message.

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset  
import os
import pandas as pd
import re

# Loading the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

vader_dataset = "vader2/vader_languages.csv"
df = pd.read_csv(vader_dataset)

# Removing the following columns to only get the ID, CWE, severity score, explanation and programming language.
df = df.drop(columns=['Unnamed: 0', 'Case', 'Repository', 'Submitted At', 'Approved At', 'num_files', 'num_languages'])

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
            row_match = df[df["ID"] == case_id]
            if not row_match.empty:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        diff = f.read()
                    df.loc[df["ID"] == case_id, "diff"] = diff
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

train_ds = dataset["train"].map(tokenizing, remove_columns=dataset["train"].column_names)
eval_ds = dataset["test"].map(tokenizing, remove_columns=dataset["test"].column_names)

# Initialize the model for LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],  
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Function for showing evaluation
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Shift labels so -100 (masked labels) are ignored
    predictions = np.argmax(logits, axis=-1)

    # Mask out -100
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    acc = accuracy.compute(predictions=predictions, references=labels)
    return {"accuracy": acc["accuracy"]}

# The training process
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results_fine_tuning",
    per_device_train_batch_size=1,    
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   
    num_train_epochs=3,
    learning_rate=2e-5,             
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=100,
    logging_dir="./logs",
    fp16=True,                    
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
