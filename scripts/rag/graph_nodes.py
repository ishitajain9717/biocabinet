"""LangGraph node functions for the interactive RAG Q&A subgraph.

Graph shape:
    START → collect_query → _route_quit → END
                                 ↓ (not quit)
                            rag_answer → collect_query  (loop)

Nodes
-----
graph_node_collect_query
    Prompt the user for a free-text biology question.  If the user
    types nothing, "quit", "exit", or "q", sets should_quit=True so
    the router exits the loop.

graph_node_rag_answer
    Retrieve relevant pathway documents for the question (with optional
    gene_filter derived from pipeline_context) and produce a grounded
    answer with [N] citations.  Uses the same LLM provider plumbing as
    the bulk pipeline (OLLAMA_MODEL / OPENAI_API_KEY, fallback to
    structured list).

Router
------
_route_after_query(state) → "rag_answer" | END
    Read state["should_quit"] and decide whether to continue or stop.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from scripts.rag.graph_state import RagChatState


# ---------------------------------------------------------------------------
# node 1 — collect the user's question
# ---------------------------------------------------------------------------

_QUIT_SIGNALS = {"", "quit", "exit", "q", "done", "bye"}


def graph_node_collect_query(state: RagChatState) -> dict:
    """Ask the user for a biology question.  Empty input → exit."""
    print("\n" + "─" * 60)
    print("RAG Q&A  (type a biology question, or press Enter to exit)")
    print("─" * 60)
    raw = input("Your question: ").strip()

    if raw.lower() in _QUIT_SIGNALS:
        return {"should_quit": True}

    return {
        "should_quit": False,
        "messages":    [HumanMessage(content=raw)],
    }


# ---------------------------------------------------------------------------
# node 2 — retrieve + answer
# ---------------------------------------------------------------------------

def graph_node_rag_answer(state: RagChatState) -> dict:
    """Retrieve relevant pathway docs and answer the latest question."""
    # Pull the last HumanMessage as the question
    question = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        return {"messages": [AIMessage(content="(no question received)")]}

    # Build gene_filter from pipeline context if available
    ctx = state.get("pipeline_context") or {}
    gene_filter: list[str] = []

    # Try to read DEG gene IDs from the significant gene TSV path
    deg_sig = ctx.get("deg_sig_path")
    if deg_sig:
        try:
            from scripts.rag.augment import _read_all_deg_ids_sorted, _select_genes
            all_genes = _read_all_deg_ids_sorted(deg_sig)
            gene_filter = _select_genes(all_genes, pinned=set())
        except Exception:
            pass

    # Call the RAG answerer
    try:
        from scripts.rag.answerer import answer
        result = answer(
            question=question,
            gene_filter=gene_filter or None,
            k=8,
            n_deg=ctx.get("n_deg_significant"),
            llm_summary=ctx.get("child_summary"),
        )

        if result.ok and result.answer:
            text = result.answer
            if result.citations:
                text += "\n\nCitations:"
                for c in result.citations:
                    text += (
                        f"\n  [{c['rank']}] {c['pathway_name']}"
                        f"  ({c['source']}, score={c['score']:.3f})"
                    )
        else:
            text = result.error or "No relevant documents found."

    except Exception as exc:
        text = f"[RAG error: {exc}]"

    print(f"\nAnswer:\n{text}\n")
    return {"messages": [AIMessage(content=text)]}


# ---------------------------------------------------------------------------
# conditional router
# ---------------------------------------------------------------------------

def _route_after_query(state: RagChatState) -> str:
    """If should_quit is True, go to END; otherwise ask + answer."""
    return "__end__" if state.get("should_quit") else "rag_answer"
