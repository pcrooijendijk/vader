import json

with open("vader2/results.json", "r") as f: 
    dataset = json.load(f)

for i in dataset: 
    print("GENERATED ANSWER")
    print(dataset[i]['generated_answer'])
    print("---------------------\n")