import json
from datasets import Dataset  
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

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
    model="./results_fine_tuning/checkpoint-39",  # path to your fine-tuned model
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = eval_ds[0]['diff']

outputs = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)

print(outputs[0]["generated_text"])
