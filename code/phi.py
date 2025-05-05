import transformers
import pandas as pd
from utils import  strip_diacritics

pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)


def auto_diacritize(text):
    messages = [
        {"role": "system", "content": "You are a linguist!"},
        {"role": "user", "content": f"Add diacritics and accent marks to this text: \n {text}"},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=128,
    )
    return outputs[0]["generated_text"][-1]['content']

df = pd.read_csv("/l/users/hawau.toyin/NLP804/_evaluation/closed_models_arabic.txt", sep="\t")
df['normalized'] = df['reference'].apply(strip_diacritics)


df['response'] = df['normalized'].apply(lambda x: auto_diacritize(x))

df.to_csv("../_results_arabic/test_phi.csv", sep="\t", index=False)
