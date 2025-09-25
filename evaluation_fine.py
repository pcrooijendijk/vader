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

# print(train_ds[0])
print(eval_ds[0])


# Reload model with LoRA 
pipe = pipeline(
    "text-generation",
    model="./results_fine_tuning/checkpoint-39",  
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = eval_ds[0]['diff']

outputs = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)

print(outputs[0]["generated_text"])
