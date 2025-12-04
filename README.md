## Colab QLoRA Fine-Tuning Toolkit

This repo provides a fully documented, Colab-ready workflow for fine-tuning open-source chat models such as `meta-llama/Llama-2-7b-chat-hf` using QLoRA on private text data. It expands upon the original Polo Club tutorial by adding richer explanations, guardrails, and extensibility tips so you can adapt it to new domains or larger models without guesswork.

### What's inside

- `notebooks/colab_qlora_finetune.ipynb` — the main Colab notebook that automates data prep, QLoRA training, evaluation, and export.
- `README.md` (this file) — deep-dives on prerequisites, how to customize the notebook, and troubleshooting tips.

If you just want to run the pipeline, open the notebook directly in Colab (File ▸ Open Notebook ▸ GitHub) or use:

```
https://colab.research.google.com/github/<your-org-or-username>/Auto-RL/blob/main/notebooks/colab_qlora_finetune.ipynb
```

Replace `<your-org-or-username>` with the account that hosts this repository.

---

## 1. Prerequisites

| Requirement | Purpose | Notes |
|-------------|---------|-------|
| Google account with Colab access | Compute | Works with the free T4 runtime (16 GB). Pro/Pro+ lets you select L4/A100 for faster runs. |
| Hugging Face account | Model + token | Request access to `meta-llama/Llama-2-7b-chat-hf` (or the model you plan to fine-tune). Generate a User Access Token with *read* access. |
| Private dataset in plain text | Training data | One document per `.txt` file is easiest. The notebook includes a helper to build JSON/Parquet datasets if needed. |
| ~12 GB Google Drive space | Outputs | Stores the LoRA adapters and tokenizer snapshots. Delete old runs or export to Hugging Face Hub once finished. |

---

## 2. How the notebook is structured

1. **Environment bootstrap**
   - Installs pinned versions of `transformers`, `peft`, `trl`, `bitsandbytes`, and `accelerate`.
   - Validates GPU availability and bnb (4-bit) compatibility.
2. **Data ingestion + cleaning**
   - Accepts either: (a) text files uploaded to Colab, (b) a Google Drive folder, or (c) a Hugging Face dataset name.
   - Normalizes whitespace, removes boilerplate, and optionally chunks long passages.
3. **Prompt templating**
   - Implements system/instruction/response formatting derived from the chat template exposed by the tokenizer config (no hard-coded strings).
   - Supports multi-turn JSON transcripts if available.
4. **QLoRA training**
   - Loads base model in 4-bit NF4, freezes the base, and attaches LoRA adapters to attention and MLP modules.
   - Uses `trl.SFTTrainer` so you only need to configure Trainer arguments.
5. **Evaluation + smoke tests**
   - Computes perplexity on a held-out slice and runs a few interactive inference prompts.
6. **Export + deployment**
   - Saves adapters locally, merges (optional) for full-precision export, and can upload to the Hugging Face Hub or your Drive.

Each section has collapsible "Why this matters" callouts to explain trade-offs and default values.

---

## 3. Customizing the workflow

### 3.1 Selecting models

- Swap `model_name` in the configuration cell for any causal decoder with chat capabilities (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- Adjust `target_modules` to match the architecture (the notebook provides maps for LLaMA, Mistral, Phi-2, etc.).
- Larger models (13B/70B) need Colab Pro+ or Gradient/AWS for more VRAM; otherwise stay with 7B.

### 3.2 Data format tips

| Scenario | Recommended Format | Validation |
|----------|-------------------|-----------|
| Simple Q&A | CSV with `instruction` / `response` columns | The notebook auto-splits into prompt/answer pairs. |
| Multi-turn chat | JSON Lines, one object per conversation | Provide `system`, `messages` array, and `role` fields. |
| Long documents | Plain text files | Enable the `smart_chunking` flag to break into ~1k token chunks. |

Use the provided `inspect_dataset()` helper to preview random rows before training.

### 3.3 Hyperparameters worth tuning

| Setting | Default | When to change |
|---------|---------|----------------|
| `lr` | `2e-4` | Lower to `1e-4` for noisier data, increase slightly for tiny datasets. |
| `micro_batch_size` | `4` (tokens: 2k) | Reduce if you hit OOM; gradient accumulation keeps global batch size stable. |
| `lora_alpha` | `16` | Increase to `32` for higher-rank adapters or more aggressive updates. |
| `warmup_ratio` | `0.03` | Set to `0.1` when training for fewer than 3 epochs. |

The notebook exposes these through a single configuration dictionary so you can keep experiment logs consistent.

---

## 4. Validation and safety recommendations

1. **Hold-out set** – Reserve at least 5% of your data to compute perplexity and spot overfitting.
2. **Behavior audit** – Use the included evaluation prompts plus your domain-specific checks (toxicity, bias, leakage).
3. **System prompt locking** – Provide a clear system message describing on/off-topic guidance; update it alongside your data.
4. **Versioning** – Tag each run with a semantic version (`v1.0.0-private-law-firm`) and save config snapshots to Drive or DVC.

---

## 5. Troubleshooting quick reference

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` even at batch size 1 | Colab assigned a K80 (no 4-bit) | Rerun runtime allocation or switch to Colab Pro to force T4/L4. |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Dataset produced empty outputs | Use `validate_rows=True` to drop incomplete prompt/response pairs. |
| Training freezes around step 0 | BitsAndBytes not initialized | Re-run the setup cell to install `bitsandbytes==0.43.0` before importing `AutoModel`. |
| Adapters fail to merge | Incompatible base model | Ensure `base_model` string matches the checkpoint used during training before calling `merge_and_unload()`. |

---

## 6. Next steps

1. Duplicate the notebook and adjust the config for your data.
2. Run the provided sanity-check evaluation prompts to verify the model behaves as expected.
3. Upload adapters to a private Hugging Face repo for deployment in an inference stack (Text Generation Inference, vLLM, or Modal).


