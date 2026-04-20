# Experiment Timeline (Chronological)

This section tells the **story** of the work: intent, what was built, and the qualitative outcome. Numbers are summarized in [04_results.md](04_results.md).

## Phase A — Baseline characterization

**Intent:** Establish System 3 performance on the 100-case set before structural changes.

**Actions:**

- Ran `evaluation/run_system3_100.py` (with outputs eventually archived as baseline CSVs).
- Recorded metrics: modest hit rate and MRR, low section F1, moderate grounding—consistent with “retrieval misses + citation noise” failure mode.

**Artifacts:** `evaluation/results/system3_results_100_baseline.csv` (and related raw JSONL copies as archived in the project).

---

## Phase B — Experiment 7: embedding fine-tuning (first iterations)

**Intent:** Adapt `BAAI/bge-small-en-v1.5` to BNS-centric retrieval using synthetic or structured supervision.

### B1. Template-based synthetic pairs

**Idea:** Generate many `(query, positive_section_text)` pairs without an external LLM—fast and cheap.

**Implementation:** `phase3_embeddings/generate_template_synthetic.py` → `phase3_embeddings/bns_synthetic_pairs.jsonl` (legacy naming).

**Outcome:** Queries were **too templated** and unlike real citizen text. Fine-tuning did not produce convincing gains on the 100-case benchmark (and was abandoned as the main strategy).

### B2. IPC → BNS mapping pairs

**Idea:** Use a structured IPC→BNS mapping to build “meta-legal” queries (asking about the mapping) paired with BNS section text—easy to justify as statute-transition adaptation.

**Implementation:** Subfolder `phase3_embeddings/bns_mapping_pipeline/` (e.g. `build_mapping_pairs.py`, `ipc_bns_mapping.csv`) → `bns_mapping_pairs.jsonl`.

**Outcome:** Training distribution **did not match** incident-style queries in `test_cases.py`. Retrieval quality on the benchmark **did not improve** as hoped; this path was deprioritized.

### B3. Groq-generated incident queries (successful direction)

**Idea:** Use **Groq** (Llama 3.1 8B class) to generate **realistic citizen incident descriptions** per BNS section, paired with the actual section text from the corpus.

**Implementation:** `phase3_embeddings/generate_groq_synthetic.py` → `phase3_embeddings/bns_groq_synthetic_pairs.jsonl`.

**Outcome:** Training distribution aligned much better with the benchmark. After fine-tuning, rebuilding FAISS, and re-evaluation, metrics improved **materially** (hit rate, MRR, F1, grounding). This became the **preferred embedding supervision** for System 3 in this project.

---

## Phase C — Complaint dataset (`complaint-relevant-bns`) and hard negatives

**Intent:** Exploit Hugging Face data that looks like **real complaints** labeled with BNS sections.

**Actions:**

- New audit folder: `phase3_embeddings/dataset_experiments_v2/`
- Downloader/inspector scripts and samples under `dataset_experiments_v2/datasets/`
- `build_complaint_training_data.py` produced:
  - `complaint_bns_pairs.jsonl` (~1504 pairs)
  - `complaint_bns_hardneg_triplets.jsonl` (~1504 triplets with chapter-proximity hard negatives)

**Fine-tuning:** `phase3_embeddings/finetune_bge.py` gained modes `complaint_bns` and `complaint_bns_hardneg` (triplet loss for hard negatives).

**Outcome:** On internal eval splits and/or the 100-case benchmark, runs showed **severe regression** compared to the Groq-synthetic model—likely **label noise**, multi-label ambiguity, or distribution mismatch versus the hand-crafted 100 cases. **Conclusion:** pause heavy reliance on this dataset for embedding training until labels are cleaned or filtered.

**Roadmap doc (still useful):** `phase3_embeddings/dataset_experiments_v2/EXPERIMENTS_DATASET_PLAN.md` (experiments E0–E6, external benchmarks ILSIC / IL-PCSR, etc.).

---

## Phase D — Engineering hardening

### D1. Groq API reliability

**Problem:** Long 100-case runs hit **rate limits** even with one key.

**Solution:** Multiple keys in `.env` (`GROQ_API_KEY_1` … `GROQ_API_KEY_4`), loaded in `bns_comparison/config.py` as `GROQ_API_KEYS`. Rotation, cooldown, and per-key rate limiting in `bns_comparison/adapters/_ollama.py`. Pre-flight check: `phase3_embeddings/check_groq_keys.py`.

### D2. Training stack on Windows / CUDA

**Problems encountered:** Missing `accelerate`, `Trainer` expecting `optimizer.train()`, **CPU-only torch** vs GPU expectation, occasional native crashes when CUDA was misconfigured.

**Resolution:** Install compatible `accelerate`, add **optimizer compatibility shim** in `finetune_bge.py`, install **CUDA-enabled PyTorch** in the project venv, enforce GPU availability for training runs as coded, and document fallbacks.

### D3. Git hygiene

Large artifacts (checkpoints, `.safetensors`, FAISS binaries) were **accidentally committed** once; fixed by reset/re-commit, `git rm --cached`, and expanding `.gitignore`.

---

## Phase E — Reranker experiment (second stage)

**Motivation:** Embedding fine-tuning gains plateau or become unstable when data is noisy. A **cross-encoder reranker** can re-order top-k dense hits without retraining the bi-encoder.

**Implementation:** Root package `rerank_experiment/`:

- `adapter.py` — `System3RerankAdapter`: FAISS + `SentenceTransformer` + `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) + **strict allowed-section list** for the generator.
- `run_100.py` — same 100-case loop with separate output paths.

**Status:** Smoke and partial runs exist; **full 100-case completion** should be verified by checking `evaluation/results/system3_raw_results_rerank_exp.jsonl` line count and exporting CSV when done. See [06_reranker_and_future_work.md](06_reranker_and_future_work.md).

---

## Summary judgment

| Approach | Role in project |
|----------|-----------------|
| Template synthetic | Exploratory; insufficient realism |
| IPC–BNS mapping | Structured but wrong query distribution for incident benchmark |
| **Groq incident synthetic** | **Main successful embedding intervention** |
| Complaint HF data | High regression risk without cleaning; kept as future work |
| **Reranker + citation constraint** | **Promising** for precision/MRR without new embedding training |
