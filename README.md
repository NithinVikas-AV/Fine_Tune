# 🔎 Text Classification & Generation using BERT and DistilGPT2

This repository contains two Jupyter Notebooks demonstrating:

1. **Text Classification using BERT (Sequence-to-Classification)**
2. **Text Generation using DistilGPT2 (Sequence-to-Sequence)**

---

## 📁 Contents

- `BERT (seq to cls).ipynb` – Fine-tuning BERT for a classification task.
- `DITSTILGPT2 (seq to seq).ipynb` – Fine-tuning DistilGPT2 for text generation using PEFT (LoRA).

---

## 1️⃣ BERT: Sequence to Classification

### 📌 Description
This notebook fine-tunes a pretrained `BERT` model for a text classification task using Hugging Face's `transformers` and `datasets` libraries.

### 🔧 Setup

Install dependencies:
```bash
pip install transformers datasets accelerate pandas scikit-learn
```

### 🧠 Model
- **Model**: `bert-base-uncased`
- **Task**: Multi-class classification
- **Loss Function**: CrossEntropyLoss
- **Trainer**: `transformers.Trainer`

### 🏁 Training Configuration
- Learning rate: 2e-5
- Epochs: 3
- Batch size: 16
- Evaluation strategy: Epoch

### ✅ Outputs
- Fine-tuned model
- Evaluation metrics: accuracy, precision, recall

---

## 2️⃣ DistilGPT2: Sequence to Sequence (Text Generation)

### 📌 Description
This notebook fine-tunes a `distilgpt2` model for conditional text generation using [PEFT](https://github.com/huggingface/peft) with LoRA (Low-Rank Adaptation).

### 🔧 Setup

Install dependencies:
```bash
pip install transformers accelerate pandas peft sentencepiece
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyarrow
```

### ⚠️ Common Errors Encountered & Fixes

- `ImportError: _get_promotion_state`:  
  → Downgrade or reinstall `numpy`:  
  ```bash
  pip install numpy==1.24.4
  ```

- `ImportError: EncoderDecoderCache`:  
  → Upgrade `transformers`:  
  ```bash
  pip install --upgrade transformers
  ```

- `RuntimeError: tensors share memory`:  
  → Use `.save_model()` for proper saving if using `Trainer`.

### 🧠 Model
- **Model**: `distilgpt2`
- **Fine-tuning technique**: PEFT with LoRA
- **LoRA Config**:
  - r = 8
  - alpha = 32
  - dropout = 0.05
  - target_modules = `["c_proj"]`

### 📦 Dataset
You can load `.parquet` datasets directly from the Hugging Face Hub using `pandas.read_parquet` and a library like `pyarrow`.

---

## 🧪 Example Command for Training

Run inside the notebook or Python script:
```python
trainer.train()
```

---

## 🚀 Output

- Trained model weights
- Evaluation metrics for classification/generation quality
- Optionally push to Hugging Face Hub

---

## 📌 License

MIT License