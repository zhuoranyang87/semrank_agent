# Task 2: Agentic Critique-and-Refine Retrieval

## Overview

This repository contains Task 2 of a CSCE 670 group project on scientific document retrieval.

Task 2 adds a single agentic critique-and-refine pass on top of the weighted retrieval baseline (Weighted SemRank) built in Task 1. The goal is to address the open-loop limitation of the baseline: the system retrieves once and has no mechanism to self-correct if the first pass drifts semantically from the intended query facet.

Owners: Aney Kanji and Zhuoran Yang


## Task 2 Goal

Fix the open-loop limitation by letting the system:

1. Run the existing weighted retrieval to produce a ranked list and a set of selected concepts
2. Inspect the top retrieved abstracts for semantic drift or missing coverage
3. Ask an LLM to critique the first-pass concepts and identify what is missing or misaligned
4. Produce a refined concept set
5. Rerank once more using the refined concepts
6. Compare the agentic result against the weighted baseline

This is a single extra feedback pass, not a multi-round agent loop.


## How Task 2 Fits Into the Full Project

The full project addresses two limitations in the retrieval setup:

- Task 1 (Chenxing and Hao): weighted mechanism — improving concept weighting with Global Aspect Weighting (GAW)
- Task 2 (Aney and Zhuoran): agentic process — one self-correcting retrieval pass layered on top of Task 1

The project story is:

```
SemRank (base) → Weighted_SemRank (Task 1) → Weighted_SemRank + Agentic Pass (Task 2)
```

Task 2 depends on Task 1. The baseline compared against in this repo is the GAW-weighted SemRank result.


## Files

| File | Description |
|---|---|
| `SemRank_csfcube_agentic.ipynb` | Main experiment notebook. Runs weighted SemRank as baseline, then runs the Task 2 agentic critique-refine loop and compares results. |
| `agent_utils.py` | Helper functions for the agentic step: critique prompt, TAMU API call wrapper, output parsing, concept refinement logic. |
| `gen_phrase_embeddings.py` | Standalone script to precompute phrase embeddings for the CSFCube corpus using SPECTER-2. Run this once if the `.pt` embedding file is missing. |
| `CSFCube/` | Test annotation files for the CSFCube evaluation set (background, method, result aspects). |
| `gitignore.patch` | Leftover patch artifact from initial repo setup. Not needed at runtime. |


## Dataset

The confirmed working result uses the **CSFCube** dataset.

CSFCube is a multi-aspect scientific paper retrieval benchmark with three facets: background, method, and result. The notebook loads all three annotation files and builds a unified query set using relevance score >= 2 as the positive threshold. This produces 34 evaluation queries.

DORISMAE is referenced in some utility code but is not part of the confirmed Task 2 result.


## Running the Notebook

Prerequisites (same environment as Task 1 / SemRank):

- Python 3.10
- PyTorch, transformers, adapters, datasets, scikit-learn, tqdm, requests
- SPECTER-2 model weights (downloaded automatically on first run)
- The CSFCube corpus files: `specter2_corpus_with-topic-terms.json`, `corpus-enc-specter.pt`, `corpus-enc-index.pkl`, `specter2_corpus_with-topic-terms.json.phrase_idx.pkl`, `specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt`, `topic-enc-specter.pt`
- The topic classifier labels file: `classifier/labels.txt`

If the phrase embedding file is missing, run:

```
python gen_phrase_embeddings.py --data_dir ./CSFCube
```

To run the notebook:

1. Set your TAMU API key as an environment variable or in a `.env` file:
   ```
   TAMUS_AI_CHAT_API_KEY=your_key_here
   CHAT_API_PROVIDER=tamu
   ```
2. Open `SemRank_csfcube_agentic.ipynb` in Jupyter and run all cells in order.
3. The final eval cell prints the baseline vs agentic comparison table.

The debug CSV `csfcube_agent_debug.csv` is written at the end. It is excluded from version control by `.gitignore`.


## API and Model Note

This notebook uses the TAMU chat API gateway, not the OpenAI platform directly.

The confirmed working configuration:

- Endpoint: `https://chat-api.tamu.ai/api/chat/completions`
- Model: `protected.gpt-4.1-mini`
- Auth: Bearer token from the TAMU portal

Earlier runs failed due to a model name format mismatch and SSE response handling issues specific to the TAMU gateway. Both are resolved in the current notebook. The `api_call` function handles SSE responses and retries automatically.

Do not commit your API key. Use `.env` (already in `.gitignore`) or export the environment variable before launching Jupyter.


## Confirmed Result

Evaluated on CSFCube (34 queries):

```
                      R@50       R@100      Hit@50     Hit@100
weighted_gaw        0.5411      0.6931      0.9706      1.0000
weighted_gaw+agent  0.5542      0.7025      0.9706      1.0000
delta              +0.0131     +0.0094     +0.0000     +0.0000
```

The agentic critique-and-refine pass improved Recall@50 by +1.31 points and Recall@100 by +0.94 points over the weighted baseline. Hit rates were unchanged, meaning the agentic pass improved ranking depth rather than binary coverage.


## Limitations

- Only CSFCube is confirmed. DORISMAE has not been validated with the full Task 2 pipeline.
- The agentic pass is a single critique round. It does not iterate.
- The critique agent is constrained to select refined concepts from the candidate vocabulary already seen in the first-pass top results. It cannot introduce entirely new vocabulary.
- Results depend on the TAMU API being available and responding within the timeout.


## Future Work

- Run and validate Task 2 on DORISMAE
- Experiment with more than one critique-refine round
- Try larger TAMU models for the critique step
- Ablate the effect of the critique on specific query facets (background vs method vs result)
