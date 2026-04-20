# Dataset Audit and External Sources

This supplements `phase3_embeddings/dataset_experiments_v2/EXPERIMENTS_DATASET_PLAN.md` with a stable summary for reports.

## Hugging Face datasets inspected (v2 folder)

| Dataset | Role | Notes |
|---------|------|--------|
| `navaneeth005/complaint-relevant-bns` | Complaint narrative → multi-label BNS sections | Strong **style match** to citizen queries; labels can be **noisy** or overly multi-label |
| `navaneeth005/BNS_definitions` | Section titles + definitions | Corpus enrichment / augmentation, **not** enough alone for contrastive retrieval |
| `actuallySaptarshi/bns_act_2023` | Chat-style Q/A | Better for **generation** than retrieval embedding |
| `govintel-legal-dataset` | — | Reported **inaccessible** from environment at audit time |
| `Exploration-Lab/IL-PCSR` | Indian statute + precedent retrieval | **Gated** on Hugging Face |

Artifacts from inspection (samples, JSON reports) live under `phase3_embeddings/dataset_experiments_v2/datasets/` when present.

## Literature pointers (methodological)

Referenced in the v2 plan as relevant to **layperson queries vs statute text**:

- **ILSIC** — layperson query ↔ statute identification; highlights transfer issues when training distribution does not match user language.
- **IL-PCSR** — Indian statute + case law retrieval benchmark (useful once access is granted).
- **QBR-style** pipelines — question banks + contrastive learning + **hard negatives** (inspired complaint + hard-negative triplets in this repo).

These are **not** reimplemented here; they inform **why** hard negatives and realistic queries matter.

## Built artifacts from `complaint-relevant-bns`

Script: `phase3_embeddings/dataset_experiments_v2/build_complaint_training_data.py`

- `complaint_bns_pairs.jsonl` — (query, positive section text) pairs
- `complaint_bns_hardneg_triplets.jsonl` — adds **hard negatives** (e.g. same-chapter proximity)

Approximate scale documented in-repo: on the order of **1.5k** pairs/triplets (exact counts may vary if the script is re-run after upstream changes).

## Synthetic sources used in fine-tuning

| Source | Mechanism | Strength |
|--------|-----------|----------|
| IndicLegalQA | JSON Q/A pairs | General legal QA style |
| Template BNS | Rule-based queries | Fast, **low realism** |
| IPC→BNS mapping | Structured mapping queries | **Legal** but **not incident-shaped** |
| **Groq BNS incidents** | LLM-generated citizen text + section text | **Best alignment** with `test_cases.py` style |
