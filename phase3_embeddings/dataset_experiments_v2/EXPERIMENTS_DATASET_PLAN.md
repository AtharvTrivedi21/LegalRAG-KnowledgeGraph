# Embedding Fine-tuning Dataset Audit and Experiment Plan (v2)

## Why this folder exists
The previous 3 fine-tuning attempts underperformed because training query style did not reliably match real incident descriptions in ns_comparison/test_cases.py.
This folder contains:
- downloaded candidate datasets,
- quick schema/quality inspection,
- a concrete experiment sequence focused on incident-style statute retrieval.

## Downloaded + inspected datasets
Location: phase3_embeddings/dataset_experiments_v2/datasets

### 1) 
avaneeth005/complaint-relevant-bns (downloaded)
- Rows: 325
- Columns: instruction, input, output
- Shape: complaint-style narrative -> comma-separated BNS sections
- Quick quality stats:
  - average input length: ~819 chars
  - average labeled sections per row: 5.46
  - rows with subsection labels (e.g. 139(1)): 299/325
  - invalid section numbers outside 1..358: 0
  - unique base sections covered: 282
- Assessment:
  - Strong match to your target query style (citizen complaint text)
  - Likely noisy multi-label outputs in some rows; needs filtering/normalization

### 2) 
avaneeth005/BNS_definitions (downloaded)
- Rows: 358
- Columns: Section, Title, Legal Definition
- Shape: statute knowledge base, not query pairs
- Assessment:
  - Good for corpus normalization and synthetic query augmentation
  - Not sufficient alone for contrastive retrieval fine-tuning

### 3) ctuallySaptarshi/bns_act_2023 (downloaded)
- Rows: 5114
- Column: messages (chat format)
- Typical sample: user asks explanatory question, assistant gives long answer
- Quick quality stats:
  - average turns: 2
  - median user chars: 74
- Assessment:
  - Better for generation/chat tuning than retrieval embeddings
  - Can be mined for query paraphrases but requires heavy filtering

### 4) ashnasharma/govintel-legal-dataset (not accessible)
- Status: inaccessible from current environment (doesn't exist or cannot be accessed)
- Action: re-check exact repo id and visibility; may require auth or renamed dataset slug

### 5) Exploration-Lab/IL-PCSR (gated)
- Status: gated on Hugging Face
- Action: request access and authenticate HF token in environment before download

## Web findings that matter for your case
- **ILSIC (EACL Findings 2026):** layperson query statute identification corpus; shows court-fact-only training transfers poorly to layperson inputs.
- **IL-PCSR (EMNLP 2025):** Indian statute + precedent retrieval benchmark; useful later for broader retrieval robustness.
- **QBR (IJCAI 2025):** question-bank + contrastive learning + hard negatives + novice-style augmentation; closest methodological fit to your failure mode.

## Proposed experiment roadmap (retrieval-first)

### E0. Control run (no new training)
- Keep current best model + current FAISS
- Re-run full 100-case eval (Groq) for a stable baseline before new dataset mixing
- Output: baseline CSV in this cycle

### E1. Complaint-only fine-tune (highest priority)
- Data: complaint-relevant-bns
- Build pairs: (input complaint, positive=BNS section text for each labeled section)
- Label normalization:
  - convert 139(1) -> base 139 for retrieval target
  - deduplicate per query
- Add confidence filter:
  - remove rows with >8 labels (high ambiguity noise)
  - remove section labels missing in your corpus
- Goal: test pure layperson-style supervision

### E2. Complaint + hard negatives (QBR-inspired)
- For each positive section, sample 3 negatives:
  1. same chapter / nearby section numbers,
  2. top confusions from your current eval outputs,
  3. lexical lookalike headings (e.g., theft/snatching/robbery)
- Train with multiple-negatives ranking objective
- Goal: reduce near-miss citations and improve top-1/top-3 precision

### E3. Complaint paraphrase augmentation (controlled Groq)
- For each complaint input, generate 2 paraphrases in casual style
- Keep labels fixed (same section set)
- Add lexical diversity checks to avoid near duplicates
- Goal: improve robustness to writing variation without drifting from incident style

### E4. Mixed training (complaint + your Groq synthetic)
- Mix ratio (start): 70% complaint-derived + 30% Groq synthetic
- Reason: complaint data anchors realism; Groq synthetic increases coverage
- Goal: recall boost without style drift

### E5. Two-stage retrieval upgrade (if embedding gains plateau)
- Stage-1: embedding retrieval top-20
- Stage-2: lightweight re-ranker (cross-encoder) on top-20
- Goal: improve ranking quality (MRR/NDCG) when nearest-neighbor retrieval is too coarse

### E6. External benchmark expansion (after access)
- Add ILSIC and IL-PCSR once credentials/access are available
- Use them as out-of-domain validation, not immediate training dump
- Goal: verify generalization beyond your 100-case benchmark

## Execution order (recommended)
1. E1 -> 2. E2 -> 3. E3 -> 4. E4 -> 5. E5 (if needed)

## Success criteria per run
- Primary: mrr, section_f1, section_precision
- Secondary: hit_rate, grounding_score, latency impact
- Reject run if precision drops sharply while recall rises (over-citation pattern)

## Artifacts in this folder
- datasets/dataset_inspection_report.json
- datasets/*_sample.json
- this file: EXPERIMENTS_DATASET_PLAN.md

## Next implementation tasks (code)
1. Add new dataset builder script for complaint dataset normalization and pair creation.
2. Add hard-negative constructor.
3. Extend inetune_bge.py with new dataset modes:
   - complaint_bns
   - complaint_bns_hardneg
   - complaint_plus_groq
4. Run E1-E4 with identical eval protocol and compare against current best.
