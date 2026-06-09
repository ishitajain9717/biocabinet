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

from scripts.rag.retriever import DEFAULT_INDEX_DIR, Retriever, get_retriever

# ---------------------------------------------------------------------------
# result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a bioinformatics research assistant interpreting an RNA-seq"
    " experiment.\n\n"
    "You have three sources of information — use ALL of them:\n"
    "  1. The DEG list: specific genes that changed expression, with fold"
    " changes and adjusted p-values. Use your own biological knowledge"
    " about these genes to reason about affected pathways and mechanisms.\n"
    "  2. Numbered pathway documents from Reactome/KEGG that were retrieved"
    " as relevant evidence. Cite them with [N] markers.\n"
    "  3. The pipeline run summary (conditions, sample counts, etc.).\n\n"
    "Rules:\n"
    "- Lead with your biological reasoning about the named genes.\n"
    "- Cite pathway documents [N] where they support your answer.\n"
    "- For experiment-level questions (conditions, tools, counts) use the"
    " pipeline summary.\n"
    "- Be concise: 4-6 sentences. Do not hedge with 'without more"
    " information' if gene symbols are provided — reason from them."
)

_DOC_TEMPLATE = "[{rank}] {source} — {pathway_name}\n{text_excerpt}"

_USER_TEMPLATE = """\
{deg_block}\
{experiment_context}

Pathway documents retrieved for these genes:
{doc_block}

Question: {question}

Answer (use gene knowledge + cite documents with [N]):"""


def _build_deg_block(pipeline_results: "dict | None") -> str:
    """Build a leading gene list block the LLM can reason from directly."""
    if not pipeline_results:
        return ""
    deg_top = pipeline_results.get("deg_top") or []
    if not deg_top:
        return ""
    lines = [
        f"Differentially expressed genes in this experiment"
        f" ({pipeline_results.get('n_deg_significant', len(deg_top))}"
        " significant, sorted by |log2FC|):"
    ]
    for g in sorted(deg_top, key=lambda x: abs(x["log2fc"]), reverse=True):
        symbol = g.get("symbol") or g["gene_id"]
        arrow = "UP" if g["direction"] == "up" else "DOWN"
        lines.append(
            f"  {arrow:4s}  {symbol:10s}"
            f"  log2FC={g['log2fc']:+.2f}  padj={g['padj']:.1e}"
        )
    return "\n".join(lines) + "\n\n"


def _build_user_prompt(
    question: str,
    docs: list[dict],
    n_deg: int | None,
    llm_summary: str | None,
    pipeline_results: dict | None = None,
    max_chars_per_doc: int = 300,
) -> str:
    # Leading DEG block — lets the LLM reason from gene names directly
    deg_block = _build_deg_block(pipeline_results)

    # Experiment context (conditions, sample counts, inference summary)
    if pipeline_results:
        from scripts.rag.pipeline_context import format_pipeline_results_for_prompt

        experiment_context = format_pipeline_results_for_prompt(pipeline_results)
    else:
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
        deg_block=deg_block,
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

_OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_is_reachable(timeout: float = 3.0) -> bool:
    """Return True only if the Ollama HTTP server responds within `timeout` s."""
    try:
        import urllib.request

        req = urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return req.status == 200
    except Exception:
        return False


def _make_llm(temperature: float = 0.1):
    import os

    ollama_model = os.environ.get("OLLAMA_MODEL")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if ollama_model:
        if not _ollama_is_reachable():
            print(
                f"[RAG] OLLAMA_MODEL={ollama_model!r} is set but Ollama server is not "
                "reachable at localhost:11434 — falling back to no-LLM mode.\n"
                "      Start Ollama with:  ollama serve",
                flush=True,
            )
        else:
            try:
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=ollama_model,
                    temperature=temperature,
                    keep_alive=-1,  # keep model loaded between calls
                    client_kwargs={"timeout": 60},
                )
            except Exception:
                pass

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=openai_key,
                timeout=60,
            )
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
            citations.append(
                {
                    "rank": rank,
                    "id": d.get("id"),
                    "pathway_name": d.get("pathway_name"),
                    "source": d.get("source"),
                    "score": d.get("score"),
                }
            )
    return citations


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def answer(
    question: str,
    gene_filter: Optional[list[str] | set[str]] = None,
    k: int = 8,
    n_deg: Optional[int] = None,
    llm_summary: Optional[str] = None,
    pipeline_results: Optional[dict] = None,
    index_dir: Path = DEFAULT_INDEX_DIR,
    retriever: Optional[Retriever] = None,
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

    # Check doc relevance using a relative gap: if the top doc is barely
    # better than the median, retrieval is random-noise (BioBERT embeddings
    # cluster at 0.82-0.85 for ALL queries, so absolute thresholds don't work).
    scores = [d.get("score", 0.0) for d in docs]
    top_score = scores[0] if scores else 0.0
    if len(scores) >= 4:
        import statistics

        median_score = statistics.median(scores)
        low_relevance = not docs or (top_score - median_score) < 0.005
    else:
        low_relevance = not docs

    llm = _make_llm()
    import os

    model_name = os.environ.get("OLLAMA_MODEL") or os.environ.get("OPENAI_MODEL", "LLM")

    # 2a. Low / no relevance → skip docs, invoke LLM directly from gene list
    if low_relevance:
        if docs:
            print(
                f"  [RAG] Pathway similarity too low ({top_score:.2f})"
                " — asking LLM directly from gene list.",
                flush=True,
            )
        if llm is None:
            no_llm_msg = (
                "No relevant pathway documents found and no LLM configured.\n"
                "Set OLLAMA_MODEL or OPENAI_API_KEY for a direct answer."
            )
            return AnswerResult(
                question=question,
                answer=no_llm_msg,
                docs=docs,
                ok=True,
            )
        deg_block = _build_deg_block(pipeline_results)
        direct_prompt = (
            f"{deg_block}"
            f"Question: {question}\n\n"
            "Answer using your biological knowledge about these genes:"
        )
        print(f"Thinking with {model_name}...", flush=True)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            resp = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=direct_prompt),
                ]
            )
            return AnswerResult(
                question=question,
                answer=resp.content,
                citations=[],
                docs=[],
                ok=True,
            )
        except Exception as exc:
            return AnswerResult(
                question=question,
                answer=f"(LLM error: {exc})",
                ok=False,
                error=str(exc),
            )

    # 2b. Good relevance → normal RAG path with pathway docs as context
    user_prompt = _build_user_prompt(
        question=question,
        docs=docs,
        n_deg=n_deg,
        llm_summary=llm_summary,
        pipeline_results=pipeline_results,
    )

    # 3. LLM call with pathway context
    if llm is None:
        answer_text = _fallback_answer(question, docs)
    else:
        print(
            f"Thinking with {model_name}" " (may take up to 30 s on first call)...",
            flush=True,
        )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            resp = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            answer_text = resp.content
        except Exception as exc:
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
            print(
                f"  [{c['rank']}] {c['pathway_name']}"
                f"  ({c['source']}, score={c['score']})"
            )
