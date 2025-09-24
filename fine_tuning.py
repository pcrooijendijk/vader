# The ultimate goal is to fine tune code Llama on the VADER dataset where the input is a commit message and
# the model should output a "human" like explanation for this message.

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

vader_dataset = "./data/vader.csv"
df = pd.read_csv(vader_dataset)

print(df)


