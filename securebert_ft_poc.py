from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, RobertaTokenizerFast, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
import os


def predict_mask(text, tok, mdl, top_k=5, device=None):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    mdl.to(device).eval()
    inputs = tok(text, return_tensors="pt").to(device)
    mask_pos = torch.where(inputs["input_ids"] == tok.mask_token_id)[1]
    with torch.no_grad():
        logits = mdl(**inputs).logits
    out = []
    for p in mask_pos.tolist():
        top = torch.topk(logits[0, p], k=top_k).indices.tolist()
        out.append(tok.convert_ids_to_tokens(top))
    return out

def build_classification_dataset(texts, labels, tok, max_length=128):
    ds = Dataset.from_dict({"text": texts, "label": labels})
    tokfn = lambda ex: tok(ex["text"], truncation=True, padding="max_length", max_length=max_length)
    t = ds.map(tokfn, batched=True, remove_columns=["text"])
    return t.map(lambda ex: {"labels": ex["label"]}, batched=True)

def train_sequence_classifier(texts, labels, base="ehsanaghaei/SecureBERT_Plus", save_dir="./securebert-endpoint-classifier", epochs=3, batch_size=8, lr=5e-5, max_length=128):
    tok = RobertaTokenizerFast.from_pretrained(base)
    ds = build_classification_dataset(texts, labels, tok, max_length=max_length)
    cl = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    args = TrainingArguments(output_dir=save_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size, learning_rate=lr, save_total_limit=2, logging_steps=50, overwrite_output_dir=True)
    tr = Trainer(model=cl, args=args, train_dataset=ds, tokenizer=tok)
    tr.train()
    tr.save_model(save_dir)
    tok.save_pretrained(save_dir)
    return save_dir

def build_classifier_from_mlm(mlm_model, base="ehsanaghaei/SecureBERT_Plus", num_labels=2, device=None):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    cl = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels).to(device).eval()
    mlm_sd = mlm_model.state_dict()
    cl_sd = cl.state_dict()
    partial = {k: v for k, v in mlm_sd.items() if k in cl_sd and cl_sd[k].shape == v.shape}
    if partial:
        cl.load_state_dict(partial, strict=False)
    return cl

def classify_endpoints(endpoints, model=None, tokenizer_obj=None, device=None, batch_size=16):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None or tokenizer_obj is None:
        raise ValueError("Model and tokenizer must be provided or pre-trained model must be available.")

    model.to(device).eval()
    tokenizer_obj.padding_side = "right"
    out = []
    for i in range(0, len(endpoints), batch_size):
        batch = endpoints[i:i+batch_size]
        enc = tokenizer_obj(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu()
        for ep, p, prob in zip(batch, preds.tolist(), probs.tolist()):
            label = "NOT_SAFE" if p == 1 else "SAFE"
            out.append((ep, label, float(prob[p])))
    return out

if __name__ == "__main__":
    endpoints = ["http://example.com/", "http://example.com/admin/login", "http://uploads.example.com/upload.php", "http://service.example.com/api/v1/status"]
    custom_data = ["http://service.example.com/api/v1/status IS NOT SAFE"] * 5
    dataset = Dataset.from_dict({"text": custom_data})
    tokenizer = RobertaTokenizerFast.from_pretrained("ehsanaghaei/SecureBERT_Plus")
    block_size = 128

    tokenized_dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, max_length=block_size, padding="max_length"), batched=True, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(lambda ex: {"labels": ex["input_ids"]}, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    model_mlm = AutoModelForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT_Plus")

    training_args = TrainingArguments(output_dir="./securebert-cve-finetuned", overwrite_output_dir=True, num_train_epochs=3, per_device_train_batch_size=8, save_steps=500, save_total_limit=2, prediction_loss_only=True, logging_steps=100)
    trainer = Trainer(model=model_mlm, args=training_args, train_dataset=lm_dataset, data_collator=data_collator)

    trainer.train()
    trainer.save_model("./securebert-cve-finetuned")
    tokenizer.save_pretrained("./securebert-cve-finetuned")

    tokenizer_ft = RobertaTokenizerFast.from_pretrained("./securebert-cve-finetuned")
    model_ft = AutoModelForMaskedLM.from_pretrained("./securebert-cve-finetuned")
    model_cl = build_classifier_from_mlm(model_ft, base="ehsanaghaei/SecureBERT_Plus", num_labels=2)

    results = classify_endpoints(endpoints, model=model_cl, tokenizer_obj=tokenizer_ft)
    for r in results:
        print(r)




