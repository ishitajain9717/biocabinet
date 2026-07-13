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


_ALIGNMENT_KEYWORDS = (
    "alignment",
    "mapping rate",
    "mapped reads",
    "how many reads",
    "star",
    "uniquely mapped",
    "multimapped",
    "unmapped",
)

_SCRNA_CELL_KEYWORDS = (
    "how many cells",
    "cell count",
    "number of cells",
    "cells after",
    "cells surviving",
    "how many genes",
    "gene count",
    "number of genes",
    "genes after",
)

_SCRNA_CLUSTER_KEYWORDS = (
    "how many clusters",
    "cluster count",
    "number of clusters",
    "leiden",
    "clusters found",
    "clusters identified",
)

_SCRNA_MARKER_KEYWORDS = (
    "marker gene",
    "top gene",
    "cluster marker",
    "which genes",
    "what genes",
    "genes for cluster",
    "genes in cluster",
    "marker for",
    "markers for",
)

_QC_ARTIFACT_KEYWORDS = (
    "fastqc",
    "quality score",
    "adapter",
    "duplication",
    "qc report",
    "trimming",
    "which samples failed",
    "samples dropped",
    "qc gate",
)

_GNN_KEYWORDS = (
    "gnn",
    "model performance",
    "f1 score",
    "precision",
    "recall",
    "ppi model",
    "interaction model",
    "protein interaction model",
)


def _answer_from_metadata(
    question: str,
    ctx: dict,
    pr: dict | None,
    arts: dict | None = None,
    exp_summary: str = "",
) -> str | None:
    """Answer run-level questions (DEG counts, pipeline steps, databases,
    alignment stats, QC) from context and artifact scan.

    Returns a plain-text answer, or None to fall through to pathway RAG.
    """
    q = question.lower()
    pr = pr or {}
    arts = arts or {}
    data_type = pr.get("data_type", "bulk_rnaseq")

    # ── scRNA-specific metadata answers ──────────────────────────────────────
    if data_type == "scrna":
        if any(kw in q for kw in _SCRNA_CELL_KEYWORDS):
            n_cells = pr.get("n_cells")
            n_genes = pr.get("n_genes")
            parts = []
            if n_cells is not None:
                parts.append(f"Cells surviving QC + filtering: {n_cells:,}")
            if n_genes is not None:
                parts.append(f"Genes surviving filtering: {n_genes:,}")
            return "\n".join(parts) if parts else "Cell/gene counts are not available."

        if any(kw in q for kw in _SCRNA_CLUSTER_KEYWORDS):
            n_cl = pr.get("n_clusters")
            if n_cl is not None:
                return f"Leiden clustering identified {n_cl} clusters."
            return "Cluster count is not available for this run."

        if any(kw in q for kw in _SCRNA_MARKER_KEYWORDS):
            markers = pr.get("markers_top") or []
            n_cl = pr.get("n_clusters_with_markers")
            if not markers:
                return "Marker gene data is not available for this run."
            lines = [f"Top marker genes across {n_cl} cluster(s):"]
            current_cluster = None
            for m in markers:
                if m["cluster"] != current_cluster:
                    current_cluster = m["cluster"]
                    lines.append(f"  Cluster {current_cluster}:")
                direction = "▲" if m["logfoldchange"] >= 0 else "▼"
                lines.append(
                    f"    {direction} {m['gene']}"
                    f"  log2FC={m['logfoldchange']:+.2f}"
                    f"  padj={m['pval_adj']:.2e}"
                )
            return "\n".join(lines)

    # ── bulk-specific metadata answers ───────────────────────────────────────

    # --- alignment / mapping stats ---
    if any(kw in q for kw in _ALIGNMENT_KEYWORDS):
        samples = arts.get("samples") or {}
        if samples:
            lines = ["Alignment statistics (STAR):"]
            for sname, s in samples.items():
                star = s.get("star") or {}
                pct = star.get("pct_uniquely_mapped")
                n = star.get("n_uniquely_mapped")
                n_in = star.get("n_input_reads")
                multi = star.get("pct_multimapped")
                if pct is not None:
                    lines.append(
                        f"  {sname}: {pct}% uniquely mapped"
                        + (f"  ({n:,} / {n_in:,} reads)" if n and n_in else "")
                        + (f"  multi={multi}%" if multi else "")
                    )
                else:
                    lines.append(f"  {sname}: alignment stats not available")
            return "\n".join(lines)
        return "Alignment statistics are not available for this run."

    # --- FastQC / QC gate artifact questions ---
    if any(kw in q for kw in _QC_ARTIFACT_KEYWORDS):
        samples = arts.get("samples") or {}
        failed = arts.get("failed_samples") or []
        lines = []
        if failed:
            lines.append(f"Samples dropped by QC gate: {', '.join(failed)}")
        if samples:
            lines.append("Per-sample QC summary:")
            for sname, s in samples.items():
                fqc = s.get("fastqc") or {}
                mods = fqc.get("modules") or {}
                ada = mods.get("Adapter Content", "?")
                qual = mods.get("Per base sequence quality", "?")
                dup = fqc.get("pct_duplicates")
                lines.append(
                    f"  {sname}: quality={qual}  adapter={ada}"
                    + (f"  duplicates={dup}%" if dup else "")
                )
        if not lines:
            return "FastQC results are not available for this run."
        return "\n".join(lines)

    # --- GNN model performance ---
    if any(kw in q for kw in _GNN_KEYWORDS):
        if arts.get("gnn_f1") is not None:
            return (
                f"GNN PPI model performance on the test set:\n"
                f"  F1        = {arts['gnn_f1']:.3f}\n"
                f"  Precision = {arts.get('gnn_precision', '?'):.3f}\n"
                f"  Recall    = {arts.get('gnn_recall', '?'):.3f}\n"
                f"  Test size = {arts.get('gnn_n_test', '?')} pairs"
            )
        return "GNN model metrics are not available for this run."

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


# ---------------------------------------------------------------------------
# node 0 — startup: LLM infers experiment type, confirms with user
# ---------------------------------------------------------------------------

_INFER_SYSTEM_PROMPT_BULK = (
    "You are a bioinformatics assistant. You have been given a report of all "
    "artifacts produced by a bulk RNA-seq pipeline run. Based on this data, "
    "characterise the experiment in 4–6 bullet points covering:\n"
    "  • Experiment type (bulk RNA-seq, paired-end / single-end)\n"
    "  • Comparison (treated vs control, conditions)\n"
    "  • Data quality (alignment rate, adapter contamination, DEG count)\n"
    "  • Downstream results (PPI model performance if available)\n"
    "  • Any notable issues (failed samples, low mapping, etc.)\n\n"
    "Be concise and factual. Only state what the data shows. "
    "Do not speculate beyond the numbers."
)

_INFER_SYSTEM_PROMPT_SCRNA = (
    "You are a bioinformatics assistant. You have been given a report of all "
    "artifacts produced by a single-cell RNA-seq (scRNA-seq) pipeline run. "
    "Based on this data, characterise the experiment in 4–6 bullet points covering:\n"
    "  • Experiment type (scRNA-seq, dataset used)\n"
    "  • Cell and gene counts after QC and filtering\n"
    "  • Number of Leiden clusters identified\n"
    "  • Top marker genes per cluster (if available)\n"
    "  • Any notable issues (high mitochondrial %, low cell counts, etc.)\n\n"
    "Be concise and factual. Only state what the data shows. "
    "Do not speculate beyond the numbers."
)


def graph_node_infer_data_type(state: RagChatState) -> dict:
    """Run once at startup: LLM reads all artifacts, characterises the
    experiment, and asks the user to confirm before Q&A begins.

    Skipped if state['data_type_confirmed'] is already True (already ran).
    """
    if state.get("data_type_confirmed"):
        return {}

    arts = state.get("run_artifacts") or {}
    pr = state.get("pipeline_results") or {}
    data_type = pr.get("data_type", "bulk_rnaseq")

    # Build the data block the LLM will read
    try:
        from scripts.rag.artifact_reader import format_artifacts_for_prompt

        artifact_block = format_artifacts_for_prompt(arts)
    except Exception:
        artifact_block = "(artifact scan not available)"

    try:
        if data_type == "scrna":
            from scripts.rag.pipeline_context import format_scrna_results_for_prompt

            results_block = format_scrna_results_for_prompt(pr)
        else:
            from scripts.rag.pipeline_context import format_pipeline_results_for_prompt

            results_block = format_pipeline_results_for_prompt(pr)
    except Exception:
        results_block = ""

    data_block = f"{artifact_block}\n\n{results_block}".strip()

    print("\n" + "═" * 60)
    print("  RAG startup — reading pipeline artifacts...")
    print("═" * 60)

    system_prompt = (
        _INFER_SYSTEM_PROMPT_SCRNA
        if data_type == "scrna"
        else _INFER_SYSTEM_PROMPT_BULK
    )

    experiment_summary = ""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from scripts.rag.answerer import _make_llm

        llm = _make_llm()
        if llm is not None:
            print("  Asking LLM to characterise the experiment...", flush=True)
            resp = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=data_block),
                ]
            )
            experiment_summary = resp.content.strip()
        else:
            # Deterministic fallback summary
            if data_type == "scrna":
                lines = ["Experiment characterisation (scRNA-seq):"]
                if pr.get("n_cells") is not None:
                    lines.append(f"  • Cells after QC   : {pr['n_cells']:,}")
                if pr.get("n_genes") is not None:
                    lines.append(f"  • Genes after QC   : {pr['n_genes']:,}")
                if pr.get("n_clusters") is not None:
                    lines.append(f"  • Leiden clusters  : {pr['n_clusters']}")
                n_cl_m = pr.get("n_clusters_with_markers")
                if n_cl_m is not None:
                    lines.append(f"  • Clusters w/ markers: {n_cl_m}")
            else:
                lines = ["Experiment characterisation (bulk RNA-seq):"]
                conds = pr.get("conditions") or {}
                if conds:
                    lines.append(
                        f"  • Comparison : {conds.get('treated','?')} vs "
                        f"{conds.get('reference','?')}"
                    )
                n_ok = pr.get("n_samples_ok") or arts.get("config", {})
                if isinstance(n_ok, int):
                    lines.append(f"  • Samples completed: {n_ok}")
                if pr.get("n_deg_significant") is not None:
                    lines.append(f"  • Significant DEGs : {pr['n_deg_significant']}")
                if arts.get("gnn_f1") is not None:
                    lines.append(f"  • GNN PPI model F1 : {arts['gnn_f1']:.3f}")
            experiment_summary = "\n".join(lines)
    except Exception as exc:
        experiment_summary = f"(could not infer experiment type: {exc})"

    print(f"\n{experiment_summary}\n")
    print("─" * 60)
    raw = input("Does this match your experiment? [Y/n/edit]: ").strip().lower()

    if raw.startswith("e"):
        correction = input("Describe the correction: ").strip()
        if correction:
            experiment_summary += f"\n\n[User correction: {correction}]"
        confirmed = True
    elif raw in ("n", "no"):
        correction = input("Please briefly describe the experiment: ").strip()
        experiment_summary = correction or experiment_summary
        confirmed = True
    else:
        confirmed = True

    print()  # blank line before Q&A prompt
    return {
        "experiment_summary": experiment_summary,
        "data_type_confirmed": confirmed,
    }


def graph_node_collect_query(state: RagChatState) -> dict:
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
    arts = state.get("run_artifacts") or {}
    exp_summary = state.get("experiment_summary") or ""

    # Route 1: pipeline metadata (DEG counts, conditions, tools used)
    meta_answer = _answer_from_metadata(question, ctx, pr, arts, exp_summary)
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
