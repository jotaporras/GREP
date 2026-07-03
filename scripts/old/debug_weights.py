"""Standalone debug script for the graph weights.

Test script to determine the fault lines in the Graph Parsing Sequence
from the DataCollator to resolve edge-weight issues.

Usage:
    python scripts/debug_weights.py
"""
import json
import re
from ast import literal_eval

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prism.data.data import SpineDataCollator


def main():
    # ── Step 1: Load model ──────────────────────────────────────────────
    models = {
        'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', 896),
        'llama': ('meta-llama/Llama-3.2-3B-Instruct', 3072)
    }

    code = 'llama'

    name, emb_dim = models[code]
    model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Step 2: Load training data ──────────────────────────────────────
    with open("../data/eval/gpt_gen_formatted.json", "r") as f:
        raw = json.load(f)
    # Take the first element of the first conversation to check out that scene graph.
    first_prompt = raw[0]['conversations'][0]['content']
    scene_graph_text = re.findall(r"Scene graph:(.*)", first_prompt)[0]
    graph_dict = literal_eval(scene_graph_text)  # handles the single quotes safely

    train_dataset = load_dataset("json", data_files=["../data/eval/gpt_gen_formatted.json"], split="train")

    def _add_messages(example):
        example["messages"] = example["conversations"]
        return example

    def _tokenize_with_conversations(example):
        tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=True
        )
        tokenized["conversations"] = example["conversations"]
        tokenized["messages"] = example["messages"]
        return tokenized

    train_dataset = train_dataset.map(_add_messages)
    train_dataset = train_dataset.map(_tokenize_with_conversations)

    training_sample = train_dataset.select(range(10))
    collator = SpineDataCollator(tokenizer=tokenizer, mlm=False)

    for example in training_sample:
        pyg_graph, injection_map = collator._extract_graph(example)
        print(pyg_graph)
        print(injection_map)



if __name__ == "__main__":
    main()
