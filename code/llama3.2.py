import torch
import pandas as pd
from transformers import pipeline
from utils import strip_diacritics
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def auto_diacritize(text):
    messages = [
        {"role": "system", "content": "You are a linguist!"},
        {"role": "user", "content": f"Add diacritics and accent marks to this text: \n {text}"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]['content']

df = pd.read_csv("/l/users/hawau.toyin/NLP804/_evaluation/closed_models_arabic.txt", sep="\t")
df['normalized'] = df['reference'].apply(strip_diacritics)

df['response'] = df['normalized'].apply(lambda x: auto_diacritize(x))

df.to_csv("../_results_arabic/test_llama3.2-3B.csv", sep="\t", index=False)

