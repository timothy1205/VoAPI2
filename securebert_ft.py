import json
import os
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    RobertaTokenizerFast,
    AutoModelForSequenceClassification
)
from VoAPIGlobalData import ApiFuncList, CWEtoApiFunc

class SecureBERTEndpointScanner:
    def __init__(self, 
                 dataset_path='endpoint_cve_dataset.jsonl', 
                 model_save_dir='./securebert-cve-finetuned', 
                 base_model="ehsanaghaei/SecureBERT_Plus"):
        
        self.dataset_path = dataset_path
        self.model_save_dir = model_save_dir
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.labels = ["SAFE"] + ApiFuncList
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

    def _load_training_data(self):
        """Parses the JSONL file to create a labeled dataset for classification."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        texts = []
        labels = []
        
        all_vulnerable_endpoints = set()
        records = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for record in records:
            vulnerable_endpoints = record.get("extracted_endpoints", [])
            if not vulnerable_endpoints:
                continue

            cwe_ids = record.get("cwe_ids", [])
            api_funcs = {CWEtoApiFunc[cwe] for cwe in cwe_ids if cwe in CWEtoApiFunc}
            
            if not api_funcs:
                continue

            primary_api_func = list(api_funcs)[0]
            label_id = self.label2id.get(primary_api_func)

            if label_id is not None:
                for endpoint in vulnerable_endpoints:
                    texts.append(endpoint)
                    labels.append(label_id)
                    all_vulnerable_endpoints.add(endpoint)

        # Add SAFE examples
        for record in records:
            description_endpoints = [ep for ep in record.get("description", "").split() if 'http' in ep or '/' in ep]
            for endpoint in description_endpoints:
                if endpoint not in all_vulnerable_endpoints:
                    texts.append(endpoint)
                    labels.append(self.label2id["SAFE"])

        if not texts:
            raise ValueError("No valid data for training.")
        
        return Dataset.from_dict({"text": texts, "label": labels})

    def train(self):
        """Fine-tunes a sequence classification model."""
        if os.path.exists(self.model_save_dir):
            return

        print(f"Model not found at {self.model_save_dir}. Starting training...")
        
        dataset = self._load_training_data()
        tokenizer = RobertaTokenizerFast.from_pretrained(self.base_model)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, 
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

        training_args = TrainingArguments(
            output_dir=self.model_save_dir,
            num_train_epochs=1000,
            per_device_train_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=tokenized_dataset,
        )

        trainer.train()
        trainer.save_model(self.model_save_dir)
        tokenizer.save_pretrained(self.model_save_dir)
        print(f"Training complete. Model saved to {self.model_save_dir}")

    def scan_endpoints(self, endpoints, batch_size=16):
        """
        Ensures model exists, loads it, and classifies the provided endpoints.
        Returns the predicted vulnerability type (test_type) if any.
        """
        self.train()

        print("Loading model for inference...")
        tokenizer = RobertaTokenizerFast.from_pretrained(self.model_save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_save_dir)
        
        model.to(self.device).eval()

        results = []
        print(f"Scanning {len(endpoints)} endpoints...")
        
        for i in range(0, len(endpoints), batch_size):
            batch = endpoints[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                preds = logits.argmax(dim=-1).cpu()
            
            for ep, p, prob in zip(batch, preds.tolist(), probs.tolist()):
                label = self.id2label[p]
                confidence = float(prob[p])
                
                result = {"endpoint": ep, "confidence": confidence}
                if label != "SAFE":
                    result["test_type"] = label
                else:
                    result["test_type"] = None
                results.append(result)
                
        return results

if __name__ == "__main__":
    # Test usage
    test_endpoints = [
        "http://example.com/admin/login", 
        "http://example.com/files/upload.php", 
        "http://example.com/images/logo.png",
        "/api/v1/system/exec?cmd=ls",
        "/api/users?query=SELECT+*+FROM+users"
    ]

    scanner = SecureBERTEndpointScanner()
    scan_results = scanner.scan_endpoints(test_endpoints)
    
    for res in scan_results:
        if res.get("test_type"):
            print(f"[VULNERABLE: {res['test_type']}] {res['endpoint']} ({res['confidence']:.2f})")
        else:
            print(f"[SAFE] {res['endpoint']} ({res['confidence']:.2f})")