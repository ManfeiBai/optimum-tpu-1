import os
from random import randint

from datasets import load_dataset
from pack_dataset import pack_dataset
from transformers import AutoTokenizer


MODEL_ID = "meta-llama/Llama-2-7b-hf"

def preprocess_dolly15k(dataset_path):
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return
    # Load dataset from the hub
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    def format_dolly(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        return prompt

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # template dataset to add prompt to each sample
    def template_dataset(sample):
        sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
        return sample

    # apply prompt template per sample
    dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
    # print random sample
    print(dataset[randint(0, len(dataset))]["text"])

    # tokenize dataset
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    )

    # chunk dataset
    lm_dataset = pack_dataset(dataset, chunk_length=2048) # We use 2048 as the maximum length for packing
    # save train_dataset to disk
    lm_dataset.save_to_disk(dataset_path)


dataset_path = "tokenized_dolly"
preprocess_dolly15k(dataset_path)
