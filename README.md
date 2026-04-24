# Task 2 Agentic Export

This folder contains only the files that were created or modified for Task 2 of CSCE 670.
It is intended to be dropped into a fresh clone of Chenxing's Weighted_SemRank repo.
The original repo files are not included here.

## Context

Task 2 adds an agentic critique-and-refine second pass on top of the existing Weighted_SemRank pipeline.
After the first-pass LLM concept selection and reranking, a critique agent inspects the top-3 retrieved
abstracts and the selected concepts, then proposes refined concepts. The pipeline reranks a second time
using those refined concepts. Baseline and agentic metrics are compared at the end.

## File inventory

### New files (did not exist in the original repo)

| File | Role |
|---|---|
| `SemRank_csfcube_agentic.ipynb` | Main Task 2 notebook. Run this end-to-end. |
| `agent_utils.py` | Helper module: critique prompt, SSE response parser, `run_critique_agent`. |
| `gen_phrase_embeddings.py` | One-time script to generate the missing phrase embedding file `specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt`. Run once before the notebook if that file is absent. |
| `.env.template` | Environment variable template. Copy to `.env` and fill in your TAMUS key. |
| `CSFCube/test-pid2anns-csfcube-background.json` | Required eval annotation file (not in original repo). |
| `CSFCube/test-pid2anns-csfcube-method.json` | Required eval annotation file (not in original repo). |
| `CSFCube/test-pid2anns-csfcube-result.json` | Required eval annotation file (not in original repo). |

### Modified files (changed from the original repo)

| File | What changed |
|---|---|
| `gitignore.patch` | Patch for `.gitignore`. Adds exclusions for `.env`, Task 2 context docs, and debug output CSVs. Apply with `git apply gitignore.patch` from the repo root. |

### Intentionally excluded

| File | Reason |
|---|---|
| `CSFCube/specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt` | 159 MB generated file. Run `gen_phrase_embeddings.py` to regenerate it in a new environment. |
| `.env` | Contains a real API key. Use `.env.template` instead and paste your own key. |
| All other original Weighted_SemRank files | Unchanged from Chenxing's repo. Clone that repo separately and overlay these files. |

## Setup in a new repo

1. Clone or copy the original Weighted_SemRank repo.
2. Copy these files into the repo root (and `CSFCube/` subfolder) preserving the paths shown above.
3. Apply the gitignore patch:
   ```
   git apply gitignore.patch
   ```
4. Copy `.env.template` to `.env` and paste your TAMUS API key:
   ```
   CHAT_API_PROVIDER=tamu
   TAMUS_AI_CHAT_API_KEY=<your-key>
   TAMUS_AI_CHAT_API_ENDPOINT=https://chat-api.tamu.ai
   TAMU_DEFAULT_MODEL=protected.gemini-2.0-flash-lite
   ```
5. If `CSFCube/specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt` is missing, run:
   ```
   python gen_phrase_embeddings.py
   ```
6. Open `SemRank_csfcube_agentic.ipynb` in Jupyter with a Python 3.11 kernel and run top to bottom.

## Environment expectations

- Python 3.11.5 kernel
- All dependencies from the original Weighted_SemRank repo (torch, transformers, adapters, etc.)
- TAMUS AI Chat API key with access to `protected.gemini-2.0-flash-lite`
- MPS (Apple Silicon) or CUDA GPU for embedding generation; CPU works but is slow
