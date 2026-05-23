"""Phase 3 of the RAG layer: retrieve + LLM synthesis with citations.

Public API
----------
    from scripts.rag.answerer import answer

    result = answer(
        question    = "Why is CDK4 upregulated in tumour samples?",
        gene_filter = ["CDK4", "CCND1", "RB1"],    # DEG gene symbols or IDs
        k           = 8,
        n_deg       = 42,                          # optional extra context
        llm_summary = "...",                       # optional pipeline summary
    )

    result.answer           # full prose answer with [N] citation markers
    result.citations        # list of {"rank":N, "id":..., "pathway_name":...,
                            #          "source":..., "score":...}
    result.docs             # the raw retrieved docs (list[dict])
    result.ok               # True unless retrieval failed entirely

DESIGN
------
    1. retrieve(question, k, gene_filter)  →  top-k docs
    2. Build a prompt that gives the LLM:
         - the user question
         - numbered doc excerpts (pathway_name + first 300 chars of text)
         - optional context (deg count, pipeline summary)
    3. Call LLM (ChatOllama or ChatOpenAI via env vars).
       Fallback: if no LLM available, return a structured text-only answer
       built from doc titles + scores (no hallucination, fully reproducible).
    4. Parse [1] … [N] citations from the answer text.
    5. Return AnswerResult.

Environment variables for LLM selection (same as bulk pipeline):
    OLLAMA_MODEL      if set, use ChatOllama (local)  e.g. "llama3"
    OPENAI_API_KEY    if set, use ChatOpenAI
    (Ollama takes priority over OpenAI)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scripts.rag.retriever import Retriever, get_retriever, DEFAULT_INDEX_DIR


# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    question:  str
    answer:    str
    citations: list[dict]         = field(default_factory=list)
    docs:      list[dict]         = field(default_factory=list)
    ok:        bool               = True
    error:     Optional[str]      = None


# ---------------------------------------------------------------------------
# prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a bioinformatics research assistant. "
    "Answer the question below using ONLY the provided pathway documents as evidence. "
    "Cite sources using [N] markers that correspond to the numbered documents. "
    "If the documents do not contain sufficient information, say so honestly. "
    "Be concise: 3–6 sentences maximum."
)

_DOC_TEMPLATE = "[{rank}] {source} — {pathway_name}\n{text_excerpt}"

_USER_TEMPLATE = """\
Context about this RNA-seq experiment:
{experiment_context}

Relevant pathway documents:
{doc_block}

Question: {question}

Answer with inline citations [N]:"""


def _build_user_prompt(
    question:   str,
    docs:       list[dict],
    n_deg:      int | None,
    llm_summary: str | None,
    max_chars_per_doc: int = 300,
) -> str:
    # experiment context block
    ctx_parts: list[str] = []
    if n_deg is not None:
        ctx_parts.append(f"- Number of differentially expressed genes: {n_deg}")
    if llm_summary:
        ctx_parts.append(f"- Pipeline summary: {llm_summary[:400]}")
    experiment_context = "\n".join(ctx_parts) if ctx_parts else "(none provided)"

    # numbered doc block
    doc_lines = []
    for doc in docs:
        excerpt = doc.get("text", "")[:max_chars_per_doc].replace("\n", " ").strip()
        doc_lines.append(
            _DOC_TEMPLATE.format(
                rank=doc["rank"],
                source=doc.get("source", "?"),
                pathway_name=doc.get("pathway_name", ""),
                text_excerpt=excerpt,
            )
        )
    doc_block = "\n\n".join(doc_lines)

    return _USER_TEMPLATE.format(
        experiment_context=experiment_context,
        doc_block=doc_block,
        question=question,
    )


# ---------------------------------------------------------------------------
# deterministic fallback (no LLM)
# ---------------------------------------------------------------------------

def _fallback_answer(question: str, docs: list[dict]) -> str:
    """Build a structured answer from doc titles alone — no LLM needed."""
    lines = [
        f"Based on {len(docs)} retrieved pathway documents, "
        "the following pathways are most relevant to your question:\n"
    ]
    for d in docs:
        lines.append(
            f"  [{d['rank']}] {d['pathway_name']} "
            f"({d['source']}, similarity={d['score']:.3f})"
        )
    lines.append(
        "\n(No LLM configured — set OLLAMA_MODEL or OPENAI_API_KEY for a prose answer.)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helper (reuses the pattern from bulk pipeline graph_nodes.py)
# ---------------------------------------------------------------------------

def _make_llm(temperature: float = 0.1):
    import os
    ollama_model = os.environ.get("OLLAMA_MODEL")
    openai_key   = os.environ.get("OPENAI_API_KEY")

    if ollama_model:
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=ollama_model, temperature=temperature)
        except Exception:
            pass

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=openai_key)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# citation parser
# ---------------------------------------------------------------------------

def _parse_citations(answer_text: str, docs: list[dict]) -> list[dict]:
    """Extract [N] references from the answer and match them to docs."""
    mentioned_ranks = set(int(m) for m in re.findall(r"\[(\d+)\]", answer_text))
    citations = []
    doc_by_rank = {d["rank"]: d for d in docs}
    for rank in sorted(mentioned_ranks):
        if rank in doc_by_rank:
            d = doc_by_rank[rank]
            citations.append({
                "rank":         rank,
                "id":           d.get("id"),
                "pathway_name": d.get("pathway_name"),
                "source":       d.get("source"),
                "score":        d.get("score"),
            })
    return citations


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def answer(
    question:    str,
    gene_filter: Optional[list[str] | set[str]] = None,
    k:           int  = 8,
    n_deg:       Optional[int] = None,
    llm_summary: Optional[str] = None,
    index_dir:   Path = DEFAULT_INDEX_DIR,
    retriever:   Optional[Retriever] = None,
) -> AnswerResult:
    """Retrieve + synthesise an answer to `question`.

    Parameters
    ----------
    question    : free-text question from the user (or from the pipeline)
    gene_filter : DEG gene IDs/symbols to focus retrieval
    k           : number of docs to retrieve
    n_deg       : DEG count (optional experiment context for the LLM)
    llm_summary : bulk pipeline summary (optional experiment context)
    index_dir   : where build_index wrote its outputs
    retriever   : pass an already-constructed Retriever to avoid reloading
    """
    # 1. Retrieval
    try:
        r = retriever or get_retriever(index_dir=index_dir)
        docs = r.retrieve(question, k=k, gene_filter=gene_filter)
    except Exception as exc:
        return AnswerResult(
            question=question,
            answer="",
            ok=False,
            error=f"Retrieval failed: {exc}",
        )

    if not docs:
        return AnswerResult(
            question=question,
            answer="No relevant pathway documents were found for your query.",
            docs=[],
            ok=True,
        )

    # 2. Build prompt
    user_prompt = _build_user_prompt(
        question=question,
        docs=docs,
        n_deg=n_deg,
        llm_summary=llm_summary,
    )

    # 3. LLM call (with fallback)
    llm = _make_llm()
    if llm is None:
        answer_text = _fallback_answer(question, docs)
    else:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            resp = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            answer_text = resp.content
        except Exception as exc:
            # LLM failed — degrade gracefully
            answer_text = _fallback_answer(question, docs)
            answer_text += f"\n\n(LLM error: {exc})"

    # 4. Parse citations
    citations = _parse_citations(answer_text, docs)

    return AnswerResult(
        question=question,
        answer=answer_text,
        citations=citations,
        docs=docs,
        ok=True,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    gene_filter = ["CDK4", "CCND1", "CDKN1A", "RB1", "E2F1"]
    q = " ".join(sys.argv[1:]) or (
        "Why might CDK4 and CCND1 be upregulated in these tumour samples?"
    )
    print(f"Question: {q}\n")
    result = answer(q, gene_filter=gene_filter, k=6, n_deg=42)

    print("=== ANSWER ===")
    print(result.answer)
    print()
    if result.citations:
        print("=== CITATIONS ===")
        for c in result.citations:
            print(f"  [{c['rank']}] {c['pathway_name']}  ({c['source']}, score={c['score']})")
