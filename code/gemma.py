import torch
import pandas as pd
from utils import get_input_prompt, auto_diacritize, strip_diacritics
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv("/l/users/hawau.toyin/NLP804/_evaluation/closed_models_arabic.txt", sep="\t")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)


df['normalized'] = df['reference'].apply(strip_diacritics)

df['prompt'] = df['normalized'].apply(get_input_prompt)

df['response'] = df['prompt'].apply(lambda x: auto_diacritize(model, tokenizer, x))

df.to_csv("../_results_arabic/test_gemma.csv", sep="\t", index=False)