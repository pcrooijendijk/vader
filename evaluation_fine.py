import json
from datasets import Dataset  
from transformers import pipeline, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./results_fine_tuning/checkpoint-39")

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    device_map="auto",         # ensures real weights are loaded
    load_in_4bit=True          # or load_in_8bit=True if that’s what you used
)

model = PeftModel.from_pretrained(model, "./results_fine_tuning/checkpoint-39")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Writing the dataset to JSON file
with open("diff_dataset.json", "r") as f:
    diff_dataset = json.load(f)

# Making the training and evaluation dataset
rows = list(diff_dataset.values())
hf_dataset = Dataset.from_list(rows)
# Split 80/20 for training and validation
dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)

train_ds = dataset["train"]
eval_ds = dataset["test"]

prompt = eval_ds[0]['diff']
prompt += "\n Give an expalantion in natural language based on the above diff\n\nExplanation:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))