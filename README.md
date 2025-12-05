# LoRA & QLoRA Fine-Tuning Toolkit

This repo provides fully documented, Colab-ready workflows for fine-tuning open-source chat models such as `meta-llama/Llama-2-7b-chat-hf` using **LoRA** (Low-Rank Adaptation) and **QLoRA** (Quantized LoRA). Both notebooks include comprehensive theory explanations, practical code, and deployment guidance.

---

## 📓 What's Inside

| Notebook | Description | GPU Memory Needed |
|----------|-------------|-------------------|
| [`colab_lora_finetune.ipynb`](notebooks/colab_lora_finetune.ipynb) | Full-precision LoRA training (bf16/fp16) | ~15 GB (7B model) |
| [`colab_qlora_finetune.ipynb`](notebooks/colab_qlora_finetune.ipynb) | 4-bit quantized QLoRA training | ~6 GB (7B model) |

Both notebooks include:
- 📚 **Detailed theory explanations** with mathematical formulas
- 🔧 **Step-by-step code** with inline documentation
- 📊 **Visual diagrams** explaining the architecture
- 🚀 **Deployment guidance** for production use

---

## 🧠 LoRA vs QLoRA: Which Should I Use?

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| **Base model precision** | FP16/BF16 (16-bit) | NF4 (4-bit) |
| **Memory for 7B model** | ~15 GB | ~6 GB |
| **Training speed** | Faster | Slightly slower |
| **Quality** | Reference | ~99% of LoRA |
| **Colab compatibility** | Pro recommended | Works on free T4 |

### Choose **LoRA** when:
- You have ≥24GB VRAM available
- Maximum quality is critical
- Faster training is preferred
- Simpler setup is desired

### Choose **QLoRA** when:
- Limited GPU memory (<16GB)
- Using free Google Colab (T4)
- Training larger models (13B+)
- Memory efficiency is priority

---

## 🚀 Quick Start

### Open in Colab

**QLoRA (recommended for free Colab):**
```
https://colab.research.google.com/github/<your-username>/Auto-RL/blob/main/notebooks/colab_qlora_finetune.ipynb
```

**LoRA (for Colab Pro or local GPUs):**
```
https://colab.research.google.com/github/<your-username>/Auto-RL/blob/main/notebooks/colab_lora_finetune.ipynb
```

Replace `<your-username>` with the account hosting this repository.

---

## 📋 Prerequisites

| Requirement | Purpose | Notes |
|-------------|---------|-------|
| Google account with Colab access | Compute | Free T4 works for QLoRA; Pro/Pro+ recommended for LoRA |
| Hugging Face account | Model access | Request access to gated models (e.g., Llama-2) and generate an access token |
| Training data | Fine-tuning | Text files, JSON, or Hugging Face datasets |
| ~12 GB Google Drive space | Outputs | Stores adapters and checkpoints |

---

## 📖 Theory Overview

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:

1. **Freezes** the pre-trained model weights
2. **Injects** small trainable matrices into each layer
3. **Trains** only ~0.1% of parameters while achieving near full fine-tuning quality

**The key insight:** Weight updates during fine-tuning have low intrinsic rank, so we can decompose:

$$W_{new} = W_{frozen} + \frac{\alpha}{r} \cdot B \cdot A$$

Where $A$ and $B$ are small matrices with rank $r \ll d$.

### What is QLoRA?

**QLoRA** combines LoRA with 4-bit quantization:

1. **Quantizes** the base model to 4-bit NF4 format (75% memory savings)
2. **Applies** LoRA adapters in full precision (bf16)
3. **Dequantizes** on-the-fly during forward pass

This enables fine-tuning a **7B model on just 6GB VRAM**!

---

## 🏗️ Notebook Structure

Both notebooks follow the same structure:

### 1. Environment Setup
- Install pinned dependencies (`transformers`, `peft`, `trl`, etc.)
- Validate GPU availability
- Authenticate with Hugging Face

### 2. Data Ingestion
- Load from text files, Google Drive, or Hugging Face datasets
- Normalize and chunk long documents
- Convert to instruction/response format

### 3. Prompt Templating
- Use model's native chat template
- Format system/user/assistant messages correctly

### 4. Training
- Load base model (fp16 for LoRA, 4-bit for QLoRA)
- Attach LoRA adapters to target modules
- Fine-tune with `SFTTrainer`

### 5. Evaluation
- Compute perplexity on held-out data
- Interactive chat testing

### 6. Export & Deployment
- Save adapters separately or merge into base model
- Optional upload to Hugging Face Hub

---

## ⚙️ Configuration Guide

### Key Hyperparameters

| Parameter | LoRA Default | QLoRA Default | Description |
|-----------|--------------|---------------|-------------|
| `lora_r` | 16 | 64 | Rank of LoRA matrices (higher = more capacity) |
| `lora_alpha` | 32 | 16 | Scaling factor (effective scale = alpha/r) |
| `lora_dropout` | 0.05 | 0.1 | Dropout for regularization |
| `learning_rate` | 1e-4 | 2e-4 | Learning rate |
| `micro_batch_size` | 1 | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | 4 | Steps to accumulate |
| `cutoff_len` | 2048 | 2048 | Maximum sequence length |

### Why Different Defaults?

- **QLoRA uses higher `lora_r`** (64 vs 16): Compensates for information loss from 4-bit quantization
- **QLoRA uses larger batches** (4 vs 1): Less memory used by base model allows bigger batches
- **QLoRA uses slightly higher LR**: Works well with the quantized setup

### Target Modules

For Llama/Mistral architectures:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]
```

---

## 📊 Data Format

### Supported Formats

| Format | Example | Use Case |
|--------|---------|----------|
| Text files | `.txt` files in a folder | Documents, articles |
| Hugging Face dataset | `tatsu-lab/alpaca` | Pre-formatted instruction data |
| JSON/JSONL | `{"instruction": "...", "response": "..."}` | Custom instruction pairs |

### Required Fields

```json
{
  "instruction": "Explain quantum computing",
  "response": "Quantum computing uses quantum mechanics...",
  "system": "You are a helpful science tutor."
}
```

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Insufficient VRAM | Reduce `micro_batch_size` or `cutoff_len` |
| Loss not decreasing | LR too low or data issues | Increase `learning_rate`, check data quality |
| Poor generation quality | Wrong prompt template | Ensure template matches training format |
| `bitsandbytes` errors | Missing CUDA support | Re-run setup cell, ensure GPU runtime |
| Training freezes | BitsAndBytes initialization | Install `bitsandbytes` before importing models |

---

## 🚀 Deployment Options

### Option 1: Keep Adapters Separate
```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(model, "path/to/adapter")
```
- ✅ Hot-swap different adapters
- ✅ Smaller storage per task
- ❌ Requires PEFT at inference

### Option 2: Merge and Export
```python
merged = model.merge_and_unload()
merged.save_pretrained("merged-model")
```
- ✅ Standard model format
- ✅ No PEFT dependency
- ❌ Larger storage

### Production Frameworks

| Framework | LoRA Support | Best For |
|-----------|--------------|----------|
| vLLM | ✅ Native | High-throughput serving |
| TGI | ✅ Native | Production APIs |
| Ollama | Merge first | Local development |
| llama.cpp | Merge + GGUF | Edge/CPU deployment |

---

## 📚 Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research by Microsoft
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Quantized fine-tuning research
- [PEFT Documentation](https://huggingface.co/docs/peft) - Hugging Face implementation
- [TRL Documentation](https://huggingface.co/docs/trl) - Training library details

---

## 📝 License

This project is provided for educational purposes. Please ensure compliance with the licenses of any models you fine-tune (e.g., Llama 2 Community License).

---

*Happy fine-tuning! 🚀*
