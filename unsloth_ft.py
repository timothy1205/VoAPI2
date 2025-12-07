#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, Dict, Any, List

# Import unsloth first to patch other libraries
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MAX_SEQ_LENGTH = 2048
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
DEFAULT_OUTPUT_DIR = "training_output"
DEFAULT_DATA_DIR = "training_data"


def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a newline-delimited JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def normalize_cwe(cwe_list: List[str]) -> List[str]:
    if not cwe_list:
        return []
    return [c.strip() for c in cwe_list]

def make_example(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a prompt/completion pair.
    """
    desc = record.get("description", "").strip()
    cve_id = record.get("cve_id", "")
    cwe_ids = normalize_cwe(record.get("cwe_ids", []))
    extracted_endpoints = record.get("extracted_endpoints", []) or []

    prompt_text = (
        "Extract the CVE identifier, CWE identifiers and any endpoints mentioned "
        "from the following vulnerability description. Return only JSON.\n\n"
        "Description:\n"
        f"{desc}\n\n"
        "JSON:"
    )

    completion_obj = {
        "cve_id": cve_id,
        "cwe_ids": cwe_ids,
        "extracted_endpoints": extracted_endpoints
    }
    completion_text = json.dumps(completion_obj, ensure_ascii=False)
    
    return {"prompt": prompt_text, "completion": completion_text + "\n"}

def write_jsonl(path: Path, examples: Iterable[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

def process_dataset(input_path: str, out_dir: str, valid_frac: float = 0.02, seed: int = 42, max_examples: int = None):
    """
    Reads the raw input JSON, formats it, and saves train/valid JSONL files.
    Returns the path to the training file.
    """
    input_p = Path(input_path).expanduser()
    out_p = Path(out_dir).expanduser()
    
    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")

    print(f"--> Processing data from {input_p}...")
    
    records = list(iter_ndjson(input_p))
    if max_examples:
        records = records[:max_examples]
        
    examples = [make_example(r) for r in records]
    random.Random(seed).shuffle(examples)
    
    n = len(examples)
    n_valid = max(1, int(n * valid_frac)) if n > 0 else 0
    valid = examples[:n_valid]
    train = examples[n_valid:]

    train_path = out_p / "train.jsonl"
    valid_path = out_p / "valid.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    print(f"    Processed {n} records -> Train: {len(train)}, Valid: {len(valid)}")
    print(f"    Saved to: {out_p}")
    
    return str(train_path)


def formatting_func(example):
    """
    Formats the dataset into the Llama-3 instruction format for the SFTTrainer.
    """
    p = (example.get("prompt") or "").strip()
    c = (example.get("completion") or "").strip()
    return {
        "text": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{c}<|eot_id|>"
        )
    }


def get_model_and_tokenizer(model_dir: str, train_data_path: str = None, retrain: bool = False):
    """
    Loads an existing fine-tuned model from `model_dir` if available.
    Otherwise, trains a new one using `train_data_path`.
    """
    model_exists = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    should_train = retrain or not model_exists

    model = None
    tokenizer = None

    if should_train:
        print(f"--> Status: Training New Model (Force: {retrain}, Exists: {model_exists})")
        
        if not train_data_path or not os.path.exists(train_data_path):
             raise ValueError(f"Cannot train: Training data not found at '{train_data_path}'. Please provide --input-data.")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True,
            load_in_8bit = False,
            load_in_16bit = False,
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, 
            bias = "none",   
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = MAX_SEQ_LENGTH,
            use_rslora = False,
            loftq_config = None, 
        )

        dataset = load_dataset("json", data_files=train_data_path, split="train")
        dataset = dataset.map(formatting_func)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = MAX_SEQ_LENGTH,
            args = SFTConfig(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 10,
                max_steps = 5000,
                logging_steps = 1,
                output_dir = model_dir,
                optim = "adamw_8bit",
                seed = 3407,
                bf16=False,
                fp16=False,
            ),
        )
        
        trainer.train()
        
        print(f"Saving model to {model_dir}...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    else:
        print(f"Loading Existing Model from {model_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_dir,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, user_query, max_tokens=256, temperature=0.0):
    """
    Generates a response for a given query using the loaded model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LLAMA PROMPT FORMATTING
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample = (temperature > 0),
        temperature = temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        assistant_part = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
        assistant_part = assistant_part.split("<|eot_id|>")[0].strip()
        return assistant_part
    else:
        return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth Fine-Tuning Pipeline")
    
    parser.add_argument("--input-data", type=str, help="Path to raw NDJSON CVE dataset to process.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to save/load JSONL training data.")
    
    parser.add_argument("--model-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save/load the fine-tuned model.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if model exists.")
    
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to run inference on after loading/training.")

    args = parser.parse_args()

    train_file_path = os.path.join(args.data_dir, "train.jsonl")
    
    if args.input_data:
        train_file_path = process_dataset(
            input_path=args.input_data, 
            out_dir=args.data_dir
        )
    else:
        if not os.path.exists(train_file_path) and (args.retrain or not os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))):
            print("No training or model data!")

    try:
        model, tokenizer = get_model_and_tokenizer(
            model_dir=args.model_dir,
            train_data_path=train_file_path,
            retrain=args.retrain
        )
        
        if args.prompt:
            print(f"Input: {args.prompt}")
            response = generate_response(model, tokenizer, args.prompt)
            print("Response:")
            print(response)
        else:
            print("No prompt given, exiting...")

    except Exception as e:
        print(f"Error: {e}")