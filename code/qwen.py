import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import  strip_diacritics
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def auto_diacritize(text):
    messages = [
        {"role": "system", "content": "You are a linguist."},
        {"role": "user", "content": f"Add diacritics and accent marks to this text: \n {text}"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

df = pd.read_csv("/l/users/hawau.toyin/NLP804/_evaluation/closed_models_arabic.txt", sep="\t")
df['normalized'] = df['reference'].apply(strip_diacritics)

df['response'] = df['normalized'].apply(lambda x: auto_diacritize(x))

df.to_csv("../_results_arabic/test_qwen.csv", sep="\t", index=False)