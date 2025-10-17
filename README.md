# DistilBERT on PubMed 20k RCT — Full Fine-Tuning vs LoRA

---

## 1) Environment

### Kaggle (recommended)

- Hardware: Kaggle free-tier GPU

- Python: 3.10+ / 3.11

- Packages

  ```bash
  pip install "pyarrow>=14,<20" -q
  pip install -q \
    "transformers==4.44.2" \
    "datasets==2.19.1" \
    "peft==0.11.1" \
    "accelerate==0.33.0" \
    "scikit-learn>=1.3" \
    "safetensors>=0.4" \
    "tokenizers>=0.19,<0.21"
  ```

---

## 2) Dataset

- **Name**: PubMed 20k RCT (mirror: `armanc/pubmed-rct20k`)  

- **Splits**: train 176,642 / validation 29,672 / test 29,578  

- Loaded directly from the Hugging Face Hub:

  ```python
  from datasets import load_dataset
  dataset = load_dataset("armanc/pubmed-rct20k")
  ```

---

## 3) Model

- **Backbone**: `distilbert-base-uncased` (≈66M params, 6 layers, 12 heads, hidden size 768).  
- **Tokenizer**: DistilBERT uncased WordPiece; vocab 30,522.  

---

## 4) Fine-tuning strategies and key hyperparameters

### Preprocessing

- Tokenize `text` with `max_length=128`, `truncation=True`.
- Dynamic padding via `DataCollatorWithPadding`.
- Metrics: Accuracy and Macro-F1 (scikit-learn).

### Full FT

- Trainable params: ~66.9M (entire model + classifier).
- Optimizer: AdamW  
  Learning rate 3e-5, epochs 2, train batch 16, eval batch 32, fp16 True, weight decay 0.01.  
- No warmup, no gradient accumulation, gradient clipping off.  
- Disable intermediate checkpoints to avoid disk pressure; evaluate after training.

### LoRA

- Freeze backbone; train adapters + classifier only.  
- Targets: DistilBERT attention projections `["q_lin","v_lin"]`.  
- LoRA config: rank r=8, alpha=16, dropout=0.1.  
- Optimizer: AdamW  
  Learning rate 2e-4, epochs 2, train batch 32, eval batch 64, fp16 True.  
- Trainable params: ~0.742M (~1% of Full FT).

---

## 5) How to run
  If can not run, please use another method.
### A. Notebook workflow 

​	Directly run transformer.ipynb in jupyter notebook

### B. Script workflow

​	Run the main.py by command as follow

```
python main.py
```
