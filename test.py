# Loading the PEFT model and then also use it for evaluation: generating answers.

import os
import re
import json
from datasets import Dataset  
from datasets import load_from_disk
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-id', type=int)

args = parser.parse_args()

dataset = load_from_disk("vader2/dataset")
print(dataset["test"][args.id]["prompt"])

# try:
#     for index, datasample in enumerate(dataset["test"]):
#         inputs = tokenizer(datasample["prompt"], return_tensors="pt").to(model.device)

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=500,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )
#         prompt_dict[index] = {
#             "prompt": datasample["prompt"],
#             "expected_answer": datasample["label"],
#             "generated_answer": tokenizer.decode(outputs[0], skip_special_tokens=True),
#         }
# except torch.OutOfMemoryError as e: 
#     print(e)
#     pass

# # Getting the results in a JSON file
# with open("results.json", "w") as f: 
#     json.dump(prompt_dict, f, indent=4)