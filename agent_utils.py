"""Critique-and-refine helpers for Task 2 (agentic second pass)."""

import re


critique_instruction = '''
You are a critique agent for scientific paper retrieval.

You will receive:
1. the original user query
2. the concepts selected during the first retrieval pass
3. the top retrieved paper abstracts from the first pass
4. an Available Vocabulary list

Your job is to identify semantic drift.
Find cases where the retrieved papers match broad keywords but miss the specific technical facet required by the query.

Return your answer in the following format exactly:

<MISSING_CONCEPTS>
concept 1, concept 2, ...
</MISSING_CONCEPTS>

<MISALIGNED_CONCEPTS>
concept 1, concept 2, ...
</MISALIGNED_CONCEPTS>

<REFINED_CONCEPTS>
concept 1, concept 2, ...
</REFINED_CONCEPTS>

<RATIONALE>
short explanation
</RATIONALE>

Rules:
- MISALIGNED_CONCEPTS must be copied exactly from the First-Pass Selected Concepts list.
- REFINED_CONCEPTS must be chosen exclusively from the Available Vocabulary list. Do not invent new terms.
- REFINED_CONCEPTS should keep useful first-pass concepts and add any missing ones from the vocabulary that better match the query.
- If the first-pass concepts are already well aligned, return them unchanged in REFINED_CONCEPTS and leave MISSING_CONCEPTS and MISALIGNED_CONCEPTS empty.
'''


def run_critique_agent(api_call_fn, query_text, selected_concepts, top_docs_text,
                       candidate_vocab=None, model='gpt-4.1-mini'):
    # one critique call reusing the notebook's existing api_call
    concepts_str = ', '.join(selected_concepts) if selected_concepts else '(none)'
    prompt_parts = [
        "Original Query:",
        query_text,
        "",
        "First-Pass Selected Concepts:",
        concepts_str,
        "",
        "Top Retrieved Abstracts:",
        top_docs_text,
    ]
    if candidate_vocab:
        vocab_str = ', '.join(candidate_vocab[:120])
        prompt_parts += ["", "Available Vocabulary (REFINED_CONCEPTS must come from this list):", vocab_str]
    prompt = "\n".join(prompt_parts) + "\n"
    return api_call_fn(prompt, instruction=critique_instruction, model=model)


def extract_tag_block(text, tag):
    # pull comma list from a tagged block, return [] if missing or empty
    if not text:
        return []
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    content = match.group(1).strip()
    if not content:
        return []
    parts = []
    for chunk in content.split(','):
        chunk = chunk.strip().lower()
        if chunk:
            parts.append(chunk)
    return parts


def parse_critique_output(text):
    return {
        'missing': extract_tag_block(text, 'MISSING_CONCEPTS'),
        'misaligned': extract_tag_block(text, 'MISALIGNED_CONCEPTS'),
        'refined': extract_tag_block(text, 'REFINED_CONCEPTS'),
    }


def build_top3_abstract_block(ranked_results, corpus_with_labels, k=3):
    # format the first k abstracts as a single text block for the critique prompt
    rows = []
    for rank in range(min(k, len(ranked_results))):
        item = ranked_results[rank]
        corpusid = item[0]
        doc = corpus_with_labels.get(str(corpusid), {})
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        rows.append(f"[{rank + 1}] Title: {title}\nAbstract: {abstract}")
    return "\n\n".join(rows)


def refine_selected_concepts(selected_concepts, critique_dict, all_term2id, max_concepts=12):
    # start from baseline concepts, drop misaligned ones, append refined ones, dedupe, cap
    base = []
    for c in selected_concepts:
        base.append(c.lower().strip())

    misaligned = set()
    for c in critique_dict.get('misaligned', []):
        if c in all_term2id:
            misaligned.add(c)

    refined_in_vocab = []
    for c in critique_dict.get('refined', []):
        if c in all_term2id:
            refined_in_vocab.append(c)

    updated = []
    seen = set()

    for c in base:
        if c in misaligned:
            continue
        if c not in all_term2id:
            continue
        if c in seen:
            continue
        updated.append(c)
        seen.add(c)

    for c in refined_in_vocab:
        if c in seen:
            continue
        updated.append(c)
        seen.add(c)

    return updated[:max_concepts]
