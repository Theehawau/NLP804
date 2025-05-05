import os
import torch
import transformers
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset, load_dataset, DatasetDict

model_id = "meta-llama/Llama-3.2-1B-Instruct"
data_path = "/l/users/hawau.toyin/NLP804/_data/sentences_01_03_manifest_train.txt"
eval_data_path = "/l/users/hawau.toyin/NLP804/_data/sentences_01_03_manifest_dev.txt"
test_data_path = "/l/users/hawau.toyin/NLP804/_data/sentences_01_03_manifest_test.txt"

save_dir = f"/l/users/hawau.toyin/NLP804/_models/{Path(model_id).stem}"
os.makedirs(save_dir, exist_ok=True)

eval_path = f"/l/users/hawau.toyin/NLP804/_finetune_results/{Path(model_id).stem}"
os.makedirs(eval_path, exist_ok=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=quant_config, device_map={"":0})

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

tr_data = Dataset.from_pandas(pd.read_csv(data_path, sep="\t"))
ev_data = Dataset.from_pandas(pd.read_csv(eval_data_path, sep="\t"))

data = DatasetDict({
    "train": tr_data,
    "test": ev_data
})


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def prepare_chat_prompt(row):
    messages = [
        {"role": "system", "content": "You are a linguist!"},
        {"role": "user", "content": f"Add diacritics and accent marks to this text: \n {row['undiacritized']}"},
        {"role": "assistant", "content": f"{row['sentence']}" }
    ]
    row['prompt'] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    return row

def prepare_test_chat_prompt(row):
    messages = [
        {"role": "system", "content": "You are a linguist!"},
        {"role": "user", "content": f"Add diacritics and accent marks to this text: \n {row['undiacritized']}"},
    ]
    row['prompt'] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return row

print("Preparing data ...")
data['train'] = data['train'].map(prepare_chat_prompt)

data['test'] = data['test'].map(prepare_test_chat_prompt)

data = data.map(lambda x: tokenizer(x["prompt"]), batched=True)


model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, 
    lora_alpha=32, #16
    target_modules=[
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=transformers.TrainingArguments(
        num_train_epochs=15,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        save_steps=25,
        save_total_limit=5,
        learning_rate=3e-4, #1e-5
        fp16=True,
        logging_steps=5,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        output_dir=save_dir,
        optim="paged_adamw_8bit" #optim="adamw_torch"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# # Train
trainer.train()
trainer.save_model(f"{save_dir}/checkpoint-best")

# Evaluate
print("Load pretrained to evaluate ...")
model = AutoModelForCausalLM.from_pretrained(f"{save_dir}/checkpoint-best", quantization_config=quant_config, device_map={"":0})


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
    return outputs[0]["generated_text"][-1]['content'].replace("\n", "").replace("Here's the text with diacritics and accent marks added:","")

df = pd.read_csv("../_data/sentences_01_03_manifest_test.txt", sep="\t")

df['response'] = df['undiacritized'].apply(lambda x: auto_diacritize(x))

df.to_csv(f"{eval_path}/test.csv", sep="\t", index=False)



