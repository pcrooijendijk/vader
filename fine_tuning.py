# The ultimate goal is to fine tune code Llama on the VADER dataset where the input is a commit message and
# the model should output a "human" like explanation for this message.

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset  
import os
import pandas as pd
import re
import json

# Loading the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

vader_dataset = "vader_languages.csv"
df = pd.read_csv(vader_dataset)

# Get the description, and pre process the data points by getting only the entries with the suggested explanation template
# List is structured as index, description
dataset = {index: desc for index, desc in enumerate(df['Description']) if "Suggested Fix" in desc}
indices = dataset.keys() # Get the case numbers

# Generate the prompt inputs where the diffs are attached
diff_path = "./cases/"
diff_dataset = {}

try:
    for root, directories, files in os.walk(diff_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filepath.endswith(".patch"):
                # Search for the case number in the .patch file
                digit = int(re.search(r'\d+', filepath).group())

                # If the digit is found, then open the file and add the diff to the dataset
                if digit in indices:
                    try: 
                        with open(filepath, "r", encoding="utf-8") as f:
                            diff = f.read()
                    except Exception as e:
                        print(f"Error reading file: {str(e)}")
                        continue

                    diff_dataset[str(digit)] = {
                        'diff' : diff,                  # The diff
                        'explanation' : dataset[digit]  # The explanation for this diff/vulnerability
                    }
except FileNotFoundError:
    print(f"Directory not found: {diff_path}")

# Writing the dataset to JSON file
with open("diff_dataset.json", "w") as f:
    json.dump(diff_dataset, f, indent=4)

# Function for tokenizing the input
def tokenizing(datapoint):
    prompt = f"Patch:\n{datapoint['diff']}\n\n Explanation:\n{datapoint['explanation']}"

    # Tokenize the input and get the ids and attention mask
    tokenized = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Making the training and evaluation dataset
rows = list(diff_dataset.values())
hf_dataset = Dataset.from_list(rows)
# Split 80/20 for training and validation
dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)

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

# The training process
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="codellama-explain-tuned",
    per_device_train_batch_size=2,    
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   
    num_train_epochs=3,
    learning_rate=2e-5,             
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
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
)

trainer.train()
