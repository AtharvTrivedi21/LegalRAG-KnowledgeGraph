# IPC to BNS Mapping Dataset Pipeline

This folder contains a clean, separate pipeline to build a retrieval fine-tuning dataset from IPC->BNS section mappings.

## Goal

Create a high-precision dataset:

- `query`: IPC-style legal query (mined from IndicLegalQA or fallback template)
- `positive`: mapped BNS section text
- `negatives`: hard negatives (nearby/same-chapter BNS sections)

Output file:

- `phase3_embeddings/bns_mapping_pipeline/bns_mapping_pairs.jsonl`

## Required Input

Provide a mapping CSV at:

- `phase3_embeddings/bns_mapping_pipeline/ipc_bns_mapping.csv`

Use the template in this folder (`ipc_bns_mapping_template.csv`) and include at least:

- `ipc_section`
- `bns_section`

Optional:

- `notes`

## Build Command

Run from project root:

```bash
python -m phase3_embeddings.bns_mapping_pipeline.build_mapping_pairs
```

## Output Structure

Each line in `bns_mapping_pairs.jsonl`:

```json
{
  "query": "What is punishment for IPC section 420?",
  "positive": "Section 318 - ...full BNS text...",
  "negatives": [
    "Section 317 - ...",
    "Section 319 - ..."
  ],
  "meta": {
    "ipc_section": "420",
    "bns_section": "318",
    "query_source": "indiclegal|template"
  }
}
```

## Notes

- This pipeline does not use LLM generation.
- It is designed to be thesis-defensible: label-preserving IPC->BNS supervision.
