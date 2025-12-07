import json
import os
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    RobertaTokenizerFast,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification
)

class SecureBERTEndpointScanner:
    def __init__(self, 
                 dataset_path='endpoint_cve_dataset.jsonl', 
                 model_save_dir='./securebert-cve-finetuned', 
                 base_model="ehsanaghaei/SecureBERT_Plus"):
        
        self.dataset_path = dataset_path
        self.model_save_dir = model_save_dir
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = 128

    def _load_training_data(self):
        """Parses the JSONL file for descriptions and endpoints."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        training_texts = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("description"):
                        training_texts.append(data["description"])
                    if data.get("extracted_endpoints"):
                        training_texts.extend(data["extracted_endpoints"])
                except json.JSONDecodeError:
                    continue
        
        if not training_texts:
            raise ValueError("No valid text found in dataset.")
        
        return training_texts

    def train(self):
        """Fine-tunes the model if the save directory does not exist."""
        if os.path.exists(self.model_save_dir):
            return

        print(f"Model not found at {self.model_save_dir}. Starting training...")
        training_texts = self._load_training_data()
        
        tokenizer = RobertaTokenizerFast.from_pretrained(self.base_model)
        dataset = Dataset.from_dict({"text": training_texts})

        tokenized_dataset = dataset.map(
            lambda e: tokenizer(e["text"], truncation=True, max_length=self.block_size, padding="max_length"), 
            batched=True, 
            remove_columns=["text"]
        )
        
        lm_dataset = tokenized_dataset.map(lambda ex: {"labels": ex["input_ids"]}, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        model_mlm = AutoModelForMaskedLM.from_pretrained(self.base_model)

        training_args = TrainingArguments(
            output_dir=self.model_save_dir, 
            overwrite_output_dir=True, 
            num_train_epochs=1000, 
            per_device_train_batch_size=8, 
            save_steps=500, 
            save_total_limit=2, 
            prediction_loss_only=True, 
            logging_steps=100
        )
        
        trainer = Trainer(
            model=model_mlm, 
            args=training_args, 
            train_dataset=lm_dataset, 
            data_collator=data_collator
        )

        trainer.train()
        trainer.save_model(self.model_save_dir)
        tokenizer.save_pretrained(self.model_save_dir)
        print(f"Training complete. Model saved to {self.model_save_dir}")

    def _build_classifier_from_mlm(self, mlm_model, num_labels=2):
        """Transfers weights from the MLM model to a Sequence Classifier."""
        cl = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=num_labels).to(self.device).eval()
        mlm_sd = mlm_model.state_dict()
        cl_sd = cl.state_dict()
        partial = {k: v for k, v in mlm_sd.items() if k in cl_sd and cl_sd[k].shape == v.shape}
        if partial:
            cl.load_state_dict(partial, strict=False)
        return cl

    def scan_endpoints(self, endpoints, batch_size=16):
        """
        Ensures model exists, loads it, and classifies the provided endpoints.
        """
        self.train()

        print("Loading model for inference...")
        tokenizer = RobertaTokenizerFast.from_pretrained(self.model_save_dir)
        
        model_mlm = AutoModelForMaskedLM.from_pretrained(self.model_save_dir)
        model_cl = self._build_classifier_from_mlm(model_mlm)
        
        tokenizer.padding_side = "right"
        model_cl.to(self.device).eval()

        results = []
        print(f"Scanning {len(endpoints)} endpoints...")
        
        for i in range(0, len(endpoints), batch_size):
            batch = endpoints[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                logits = model_cl(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                preds = logits.argmax(dim=-1).cpu()
            
            for ep, p, prob in zip(batch, preds.tolist(), probs.tolist()):
                label = "VULNERABLE" if p == 1 else "SAFE"
                confidence = float(prob[p])
                results.append({"endpoint": ep, "status": label, "confidence": confidence})
                
        return results

if __name__ == "__main__":
    # Test usage
    test_endpoints = [
        "http://example.com/admin/login", 
        "http://uploads.example.com/upload.php", 
        "http://example.com/images/logo.png"
    ]

    scanner = SecureBERTEndpointScanner()
    scan_results = scanner.scan_endpoints(test_endpoints)
    
    for res in scan_results:
        print(f"[{res['status']}] {res['endpoint']} ({res['confidence']:.2f})")