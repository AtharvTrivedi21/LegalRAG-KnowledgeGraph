# Infrastructure: APIs, Training, Git

## Groq vs Ollama

- **Ollama:** local HTTP API (`OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`).
- **Groq:** cloud API, OpenAI-compatible chat; enabled when keys are present.

Shared chat entry points live in `bns_comparison/adapters/_ollama.py` (e.g. `ollama_chat` routing to Groq when configured).

## Multi-key rotation and rate limits

**Config:** `bns_comparison/config.py` collects `GROQ_API_KEY_1` … `GROQ_API_KEY_4` (and optional `GROQ_API_KEY_5`), falling back to legacy `GROQ_API_KEY`.

**Behavior:** `_ollama.py` implements:

- Per-key **RPM-style** throttling (`GROQ_RPM_LIMIT`, default 30).
- **Rotation** on rate-limit responses: mark key cooldown, advance to next key.
- If **all** keys are cooling down, **sleep** until the earliest key is usable again.
- Increased retry budget for long jobs.

This was necessary so **100-case evaluations** and **batch Groq generation** could finish without manual key swapping.

## Key check script

`phase3_embeddings/check_groq_keys.py` sends a minimal request per configured key so you can confirm all keys work before overnight runs.

## PyTorch, CUDA, and training

- Fine-tuning uses **sentence-transformers** + **Trainer** path in `finetune_bge.py`.
- **Accelerate:** required version pinned/installed per project venv after import errors.
- **Optimizer shim:** some `Trainer` versions call `optimizer.train()` / `eval()`; PyTorch optimizers do not implement those—compat shims were added at the top of `finetune_bge.py`.
- **GPU:** training was intended to run on **CUDA** after installing the CUDA wheel of PyTorch; the script exits if CUDA is unavailable (see current `finetune_bge.py`).

## Windows / shell notes

- PowerShell does **not** use `&&` like CMD/bash for chaining; use `;` or separate lines.
- Multi-line commit messages: use PowerShell here-strings or `-m` with explicit newlines—heredoc `<<EOF` is not portable on Windows shells.

## Git strategy

- **Grouped commits** by theme (evaluation, embedding, rerank, gitignore).
- **Large binaries** removed from history/index via `git reset --soft` / selective staging and `git rm --cached`.
- **`.gitignore`** extended for: model weights, checkpoints, FAISS artifacts, backups, etc.

## Evaluation robustness

- **Resume:** JSONL stores one object per case; reruns skip completed IDs.
- **Corrupt lines:** partial failures can be removed from JSONL to force re-run of specific cases.
- **`dotenv`:** `run_system3_100.py` loads `.env` so Groq keys are found without manual export.
