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

# ---------------------------------------------------------------------------
# question classifier — biology vs casual
# ---------------------------------------------------------------------------

# If the question contains ANY of these it is treated as biology-related
# and goes to the full RAG pipeline.
_BIOLOGY_SIGNALS = (
    "gene",
    "pathway",
    "protein",
    "expression",
    "cancer",
    "tumor",
    "tumour",
    "cell",
    "signaling",
    "signalling",
    "mutation",
    "rna",
    "seq",
    "dna",
    "kinase",
    "receptor",
    "transcription",
    "regulation",
    "upregulated",
    "downregulated",
    "differentially",
    "enriched",
    "enrichment",
    "deg",
    "brca",
    "tp53",
    "p53",
    "atm",
    "cdkn",
    "mtor",
    "akt",
    "ras",
    "erk",
    "apoptosis",
    "proliferation",
    "metastasis",
    "oncogene",
    "suppressor",
    "interaction",
    "ppi",
    "binding",
    "inhibit",
    "activat",
    "phospho",
    "pathway",
    "reactome",
    "kegg",
    "go term",
    "ontology",
    "which pathway",
    "what pathway",
    "affected pathway",
    "biological",
    "mechanism",
    "function",
    "role of",
    "involved in",
)

# Questions that are clearly casual / off-topic — answer directly via LLM
# without touching the pathway index at all.
_CASUAL_SIGNALS = (
    "how are you",
    "what are you",
    "who are you",
    "tell me about yourself",
    "hello",
    "hi there",
    "good morning",
    "good afternoon",
    "hey",
    "thank you",
    "thanks",
    "great",
    "awesome",
    "cool",
    "nice",
    "what time",
    "what day",
    "what is the weather",
    "what's the weather",
    "joke",
    "funny",
    "story",
    "poem",
)


def _classify_question(question: str) -> str:
    """Return 'biology', 'casual', or 'unknown'.

    - 'biology'  → run full RAG pipeline
    - 'casual'   → call LLM directly, no retrieval
    - 'unknown'  → run RAG; LLM decides if docs are useful
    """
    q = question.lower().strip()
    if any(sig in q for sig in _CASUAL_SIGNALS):
        return "casual"
    if any(sig in q for sig in _BIOLOGY_SIGNALS):
        return "biology"
    return "unknown"


# ---------------------------------------------------------------------------
# metadata question routing
# ---------------------------------------------------------------------------

# Questions about the pipeline run itself — answer from pipeline_results/context.
_DEG_KEYWORDS = (
    "how many",
    "how much",
    "number of",
    "count",
    "total",
    "differentially expressed",
    "differentially regulated",
    "significant genes",
    "sig genes",
    "deg count",
    "n deg",
    "how many genes",
    "upregulated",
    "downregulated",
    "how many samples",
    "samples completed",
    "sample count",
)

_PIPELINE_KEYWORDS = (
    "what pipeline",
    "which pipeline",
    "pipeline used",
    "pipeline was",
    "what tool",
    "which tool",
    "what software",
    "which software",
    "what workflow",
    "which workflow",
    "what steps",
    "which steps",
    "how was this",
    "how were the",
    "how did you",
    "what was run",
    "which was run",
    "what aligner",
    "which aligner",
    "how aligned",
    "what normali",
    "which normali",
)

_CONDITIONS_KEYWORDS = (
    "what conditions",
    "which conditions",
    "experimental design",
    "what comparison",
    "which comparison",
    "treated vs",
    "control vs",
    "what groups",
    "which groups",
    "experimental groups",
)

_DATABASE_KEYWORDS = (
    "which database",
    "what database",
    "pathway database",
    "which source",
    "what source",
    "kegg",
    "reactome",
    "where do the pathways",
    "pathway source",
    "pathway library",
    "index built",
    "how many pathways",
    "number of pathways",
)


def _answer_from_metadata(question: str, ctx: dict, pr: dict | None) -> str | None:
    """Answer run-level questions (DEG counts, pipeline steps, databases) from context.

    Returns a plain-text answer, or None to fall through to pathway RAG.
    """
    q = question.lower()
    pr = pr or {}

    # --- experimental conditions questions ---
    if any(kw in q for kw in _CONDITIONS_KEYWORDS):
        conds = pr.get("conditions") or {}
        if conds:
            all_conds = conds.get("all") or {}
            if all_conds:
                groups: dict[str, list] = {}
                for sample, group in all_conds.items():
                    groups.setdefault(group, []).append(sample)
                lines = ["Experimental design:"]
                for group, samples in groups.items():
                    lines.append(f"  {group}: {', '.join(samples)}")
                return "\n".join(lines)
            return (
                f"Comparison: {conds.get('treated','treated')} vs "
                f"{conds.get('reference','control')}"
            )
        return "Experimental conditions were not recorded in the current context."

    # --- pipeline methodology questions ---
    if any(kw in q for kw in _PIPELINE_KEYWORDS):
        lines = [
            "This RNA-seq pipeline ran the following steps:",
            "  1. FastQC — read quality control",
            "  2. Trim Galore — adapter trimming (optional)",
            "  3. STAR — spliced alignment to the reference genome",
            "  4. featureCounts — gene-level read quantification",
            "  5. DESeq2 (via PyDESeq2) — differential expression analysis",
            "  6. GIN-PPI GNN — protein–protein interaction inference on DEG pairs",
            "  7. RAG (BioBERT + Reactome/KEGG) — pathway-grounded Q&A",
        ]
        conds = pr.get("conditions") or {}
        if conds:
            lines.append(
                f"\nComparison: {conds.get('treated','treated')} vs "
                f"{conds.get('reference','control')}"
            )
        if pr.get("n_samples_ok") is not None:
            lines.append(f"Samples that completed successfully: {pr['n_samples_ok']}")
        return "\n".join(lines)

    # --- pathway database questions ---
    if any(kw in q for kw in _DATABASE_KEYWORDS):
        return (
            "The RAG index was built from two pathway databases:\n"
            "  • Reactome — curated human reaction and pathway data\n"
            "  • KEGG — Kyoto Encyclopedia of Genes and Genomes pathways\n\n"
            "Documents were embedded with BioBERT (dmis-lab/biobert-base-cased-v1.2) "
            "and stored as a NumPy vector index for cosine-similarity retrieval."
        )

    # --- DEG / sample count questions ---
    if not any(kw in q for kw in _DEG_KEYWORDS):
        return None

    n_deg = pr.get("n_deg_significant") or ctx.get("n_deg_significant")
    deg_top = pr.get("deg_top") or []
    summary = ctx.get("child_summary") or ""

    if "sample" in q:
        n_ok = pr.get("n_samples_ok")
        if n_ok is not None:
            return (
                f"This run had {n_ok} sample(s)"
                " complete the full preprocessing pipeline."
            )

    if n_deg is not None:
        lines = [
            f"This run found {n_deg} statistically significant differentially "
            f"expressed gene(s) (padj < threshold, |log2FC| > threshold)."
        ]
        if deg_top:
            lines.append("\nTop differentially expressed genes:")
            for g in deg_top[:5]:
                arrow = "▲" if g["direction"] == "up" else "▼"
                lines.append(
                    f"  {arrow} {g['gene_id']}"
                    f"  log2FC={g['log2fc']:+.2f}  padj={g['padj']:.2e}"
                )
        return "\n".join(lines)

    import re

    m = re.search(r"(\d+)\s+sig(?:nificant)?\s+DEG", summary, re.IGNORECASE)
    if m:
        return (
            f"Based on the pipeline summary, there were {m.group(1)} significant "
            "differentially expressed genes in this run."
        )

    if summary:
        return (
            "The exact DEG count was not recorded in the chat context for this run. "
            f"Here is the pipeline summary:\n\n{summary}"
        )

    return (
        "The DEG count is not available in the current chat context. "
        "It is recorded in deseq2_significant.tsv under your run output directory."
    )


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
        "messages": [HumanMessage(content=raw)],
    }


# ---------------------------------------------------------------------------
# node 2 — retrieve + answer
# ---------------------------------------------------------------------------


def _direct_llm_answer(question: str, pr: dict) -> str:
    """Call the LLM directly with the gene list — no pathway retrieval."""
    try:
        from scripts.rag.answerer import _SYSTEM_PROMPT, _build_deg_block, _make_llm

        llm = _make_llm()
        if llm is None:
            return "No LLM configured. Set OLLAMA_MODEL or OPENAI_API_KEY."
        import os

        model = os.environ.get("OLLAMA_MODEL") or "LLM"
        print(f"Thinking with {model}...", flush=True)
        deg_block = _build_deg_block(pr) if pr else ""
        prompt = f"{deg_block}Question: {question}\n\n" "Answer concisely:"
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return resp.content
    except Exception as exc:
        return f"(LLM error: {exc})"


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

    ctx = state.get("pipeline_context") or {}
    pr = state.get("pipeline_results") or {}

    # Route 1: pipeline metadata (DEG counts, conditions, tools used)
    meta_answer = _answer_from_metadata(question, ctx, pr)
    if meta_answer is not None:
        print(f"\nAnswer:\n{meta_answer}\n")
        return {"messages": [AIMessage(content=meta_answer)]}

    # Route 2: casual / off-topic → LLM directly, no retrieval
    q_class = _classify_question(question)
    if q_class == "casual":
        text = _direct_llm_answer(question, pr)
        print(f"\nAnswer:\n{text}\n")
        return {"messages": [AIMessage(content=text)]}

    # Route 3: biology / unknown → full RAG pipeline
    gene_filter: list[str] = []

    # Build gene filter using Entrez IDs (RAG index is keyed by Entrez).
    # pipeline_results.deg_entrez_ids is pre-translated by pipeline_context.
    entrez_ids = pr.get("deg_entrez_ids") or []
    if entrez_ids:
        gene_filter = entrez_ids
    else:
        # Fallback: ENSG IDs from sig TSV — won't hit the Entrez-keyed index
        # but retriever will fall back to full corpus automatically.
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
            n_deg=pr.get("n_deg_significant") or ctx.get("n_deg_significant"),
            llm_summary=ctx.get("child_summary"),
            pipeline_results=pr or None,
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
