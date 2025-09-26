import json

with open("vader2/results.json", "r") as f: 
    dataset = json.load(f)

for i in dataset: 
    print("GENERATED ANSWER")
    with open(f"vader2/output/codellama_output_{i}.txt", "w") as f:
        f.write(dataset[i]['generated_answer'])