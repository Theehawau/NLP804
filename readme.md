<h1 align="center">
    <strong>Are LLMs Good Contextual Text Diacritizers? An Arabic, Yoruba Case Study</strong>
</h1>

<p align="center">
    <a href="https://github.com/Theehawau/NLP804">
        <img src="https://img.shields.io/badge/View%20on%20GitHub-blue?style=for-the-badge&logo=github" alt="GitHub Link">
    </a>
    <a href="https://huggingface.co/datasets/herwoww/MultiDiac">
        <img src="https://img.shields.io/badge/Download%20Dataset-blue?style=for-the-badge&logo=huggingface" alt="GitHub Link">
    </a>
</p>


**Automatic text diacritization** is a crucial task for languages like Arabic, where diacritics play a fundamental role in disambiguating meaning and improving readability. In this study, we evaluate the effectiveness of **4 open source Large Language Models (LLMs)** and **6 closed LLMs** in contextual Arabic text diacritization. This would eliminate the need to manually diacritize texts and improve access to diacritized resources for **L2 speakers**. We additionally create **MultiDiac**, a multilingual diacritization benchmark dataset.



## Features
- **Evaluate Open source LMs**: Use code/model_name.py to evaluate opens ource models. 
    Available model_names: gemma, llama3.2, phi, qwen.
- **Fine-tune Open source using LORA**: Use code/train.py to fine-tune opens ource models.
- **Extract response and Evaluate**: Use code/extract_results.ipynb for exact output extraction and model evaluation.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- Required libraries: `torch`, `transformers`, `datasets`, `pandas`, `evaluate`, `evaluate`