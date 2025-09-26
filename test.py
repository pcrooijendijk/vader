# Loading the PEFT model and then also use it for evaluation: generating answers.

import os
import re
import json
from datasets import Dataset  
import pandas as pd
import torch

vader_dataset = "vader2/vader_languages.csv"
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
print(dataset['label'][0])