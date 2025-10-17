import os, time, random, numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

#Repro & device ---
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

#Ensure PEFT is available (for LoRA) ---
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception as e:
    print("Installing peft ...")
    !pip -q install "peft==0.11.1" --no-deps
    from peft import LoraConfig, get_peft_model, TaskType

#Load dataset (NO DOWNSAMPLING) ---
dataset = load_dataset("armanc/pubmed-rct20k")
dataset = dataset.class_encode_column("label")

print(dataset)
feat = dataset["train"].features["label"]
label_names = list(feat.names)
label2id = {n:i for i,n in enumerate(label_names)}
id2label = {i:n for i,n in enumerate(label_names)}
num_labels = len(label_names)
print("Labels:", label_names)

#Tokenizer & preprocessing (dynamic padding) ---
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
max_length = 128

def preprocess(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
    enc["labels"] = batch["label"]
    return enc

encoded = dataset.map(
    preprocess, batched=True, remove_columns=dataset["train"].column_names
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Metrics using sklearn ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def count_all(m): return sum(p.numel() for p in m.parameters())

# =========================
# Full Fine-tuning (DistilBERT)
# =========================
set_seed(42)
model_ft = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
).to(device)

args_ft = TrainingArguments(
    output_dir="/kaggle/working/distilbert_ft_pubmed20k",
    save_strategy="no",          
    evaluation_strategy="no",    
    logging_steps=200,           
    load_best_model_at_end=False,
    save_total_limit=1,          
    save_safetensors=True,
    learning_rate=3e-5,                 
    per_device_train_batch_size=16,     
    per_device_eval_batch_size=32,
    num_train_epochs=2,                 
    #logging_steps=200,
    fp16=(device=="cuda"),
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
    
)

trainer_ft = Trainer(
    model=model_ft,
    args=args_ft,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

t0 = time.time()
trainer_ft.train()
ft_time_min = round((time.time() - t0)/60, 2)

ft_val = trainer_ft.evaluate(encoded["validation"])
ft_test = trainer_ft.evaluate(encoded["test"])
print("\n[Full FT] Val:", ft_val, " Test:", ft_test, " Time(min):", ft_time_min)

preds_ft = trainer_ft.predict(encoded["test"])
y_true_ft = preds_ft.label_ids
y_pred_ft = np.argmax(preds_ft.predictions, axis=-1)
print("\n[Full FT] Confusion matrix:\n", confusion_matrix(y_true_ft, y_pred_ft))
print("\n[Full FT] Classification report:\n",
      classification_report(y_true_ft, y_pred_ft, target_names=label_names, digits=4))

# =========================
# LoRA (PEFT) on DistilBERT
# =========================
set_seed(42)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_lin","v_lin"]
)
model_lora = get_peft_model(base_model, peft_config).to(device)
model_lora.print_trainable_parameters()

args_lora = TrainingArguments(
    output_dir="/kaggle/working/distilbert_lora_pubmed20k",
    save_strategy="no",
    evaluation_strategy="no",
    logging_steps=200,
    load_best_model_at_end=False,
    save_total_limit=1,
    save_safetensors=True,
    learning_rate=2e-4,                 
    per_device_train_batch_size=32,     
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    #logging_steps=200,
    fp16=(device=="cuda"),
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
)

trainer_lora = Trainer(
    model=model_lora,
    args=args_lora,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

t0 = time.time()
trainer_lora.train()
lora_time_min = round((time.time() - t0)/60, 2)

lora_val = trainer_lora.evaluate(encoded["validation"])
lora_test = trainer_lora.evaluate(encoded["test"])
print("\n[LoRA] Val:", lora_val, " Test:", lora_test, " Time(min):", lora_time_min)

preds_lora = trainer_lora.predict(encoded["test"])
y_true_l = preds_lora.label_ids
y_pred_l = np.argmax(preds_lora.predictions, axis=-1)
print("\n[LoRA] Confusion matrix:\n", confusion_matrix(y_true_l, y_pred_l))
print("\n[LoRA] Classification report:\n",
      classification_report(y_true_l, y_pred_l, target_names=label_names, digits=4))

# Summary
def g(d, k): return round(float(d[k]), 4) if k in d else None
summary = {
    "FT_acc_val": g(ft_val, "eval_accuracy"),
    "FT_f1_val": g(ft_val, "eval_f1_macro"),
    "FT_acc_test": g(ft_test, "eval_accuracy"),
    "FT_f1_test": g(ft_test, "eval_f1_macro"),
    "FT_params_all": sum(p.numel() for p in model_ft.parameters()),
    "FT_params_trainable": sum(p.numel() for p in model_ft.parameters() if p.requires_grad),
    "FT_time_min": ft_time_min,

    "LoRA_acc_val": g(lora_val, "eval_accuracy"),
    "LoRA_f1_val": g(lora_val, "eval_f1_macro"),
    "LoRA_acc_test": g(lora_test, "eval_accuracy"),
    "LoRA_f1_test": g(lora_test, "eval_f1_macro"),
    "LoRA_params_all": sum(p.numel() for p in model_lora.parameters()),
    "LoRA_params_trainable": sum(p.numel() for p in model_lora.parameters() if p.requires_grad),
    "LoRA_time_min": lora_time_min,
}
print("\n=== Summary ===\n", summary)
