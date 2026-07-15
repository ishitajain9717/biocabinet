"""LangChain tools for the agentic RAG answer loop.

Each tool wraps one data source the LLM can call on demand:

    retrieve_pathways   — BioBERT cosine search on KEGG/Reactome index
    read_deg_table      — top DEGs (bulk) or cluster markers (scRNA) from state
    read_alignment_stats— STAR mapping rates from artifact scan
    read_cluster_markers— per-cluster marker genes from scRNA run
    read_run_summary    — experiment type, conditions, QC counts

All tools are constructed via ``build_rag_tools`` which closes over the
pipeline state so they do not need global singletons.
"""

from __future__ import annotations

from langchain_core.tools import tool


def build_rag_tools(
    pipeline_results: dict,
    run_artifacts: dict,
    pipeline_context: dict,
    experiment_summary: str,
) -> list:
    """Return a list of configured LangChain tools for one RAG session.

    Each tool is a closure over the pipeline state passed in, so the LLM
    always reads data from the current run rather than stale globals.
    """

    # ------------------------------------------------------------------ #
    # Tool 1 — pathway retrieval                                           #
    # ------------------------------------------------------------------ #

    @tool
    def retrieve_pathways(query: str, genes: str = "") -> str:
        """Search the KEGG/Reactome pathway index for relevant documents.

        Use this for questions about biological pathways, gene functions,
        molecular mechanisms, enrichment, or when you need scientific
        context to support your answer.

        query: natural-language description of what you are looking for.
        genes: comma-separated gene symbols to focus retrieval (optional).
        Returns numbered pathway excerpts you can cite with [N].
        """
        gene_list = (
            [g.strip() for g in genes.split(",") if g.strip()] if genes else None
        )
        try:
            from scripts.rag.answerer import get_retriever

            r = get_retriever()
            docs = r.retrieve(query, k=6, gene_filter=gene_list)
        except Exception as exc:
            return f"Pathway retrieval failed: {exc}"

        if not docs:
            return "No relevant pathway documents found."

        lines = []
        for d in docs:
            excerpt = (d.get("text") or "")[:280]
            lines.append(
                f"[{d['rank']}] {d.get('source','?')} — "
                f"{d.get('pathway_name','?')}\n"
                f"  relevance={d.get('score', 0):.3f}\n"
                f"  {excerpt}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    # Tool 2 — DEG / marker table                                          #
    # ------------------------------------------------------------------ #

    @tool
    def read_deg_table(top_n: int = 20) -> str:
        """Read differentially expressed genes or scRNA marker genes.

        Use this when asked about which genes changed, fold changes,
        p-values, upregulated/downregulated genes.

        top_n: number of genes to return (default 20).
        """
        pr = pipeline_results or {}
        deg_top = pr.get("deg_top") or []

        if deg_top:
            n_sig = pr.get("n_deg_significant", len(deg_top))
            subset = sorted(deg_top, key=lambda x: abs(x["log2fc"]), reverse=True)[
                :top_n
            ]
            lines = [
                f"Top {len(subset)} of {n_sig} significant DEGs "
                f"(sorted by |log2FC|):"
            ]
            for g in subset:
                symbol = g.get("symbol") or g["gene_id"]
                arrow = "▲" if g["direction"] == "up" else "▼"
                lines.append(
                    f"  {arrow} {symbol:<12} "
                    f"log2FC={g['log2fc']:+.2f}  "
                    f"padj={g['padj']:.2e}"
                )
            return "\n".join(lines)

        # scRNA fallback — top markers across clusters
        markers = pr.get("markers_top") or []
        if markers:
            lines = [f"Top {min(top_n, len(markers))} cluster marker genes:"]
            for m in markers[:top_n]:
                direction = "▲" if m["logfoldchange"] >= 0 else "▼"
                lines.append(
                    f"  Cluster {m['cluster']}: "
                    f"{direction} {m['gene']} "
                    f"log2FC={m['logfoldchange']:+.2f} "
                    f"padj={m['pval_adj']:.2e}"
                )
            return "\n".join(lines)

        return "No DEG or marker gene data available for this run."

    # ------------------------------------------------------------------ #
    # Tool 3 — STAR alignment stats                                        #
    # ------------------------------------------------------------------ #

    @tool
    def read_alignment_stats() -> str:
        """Read STAR alignment statistics: mapping rates and read counts.

        Use this for questions about sequencing quality, how many reads
        were mapped, multi-mapping rates, or alignment performance.
        """
        arts = run_artifacts or {}
        samples = arts.get("samples") or {}
        if not samples:
            return "Alignment statistics not available for this run."

        lines = ["STAR alignment statistics:"]
        for sname, s in samples.items():
            star = s.get("star") or {}
            pct = star.get("pct_uniquely_mapped")
            n = star.get("n_uniquely_mapped")
            n_in = star.get("n_input_reads")
            multi = star.get("pct_multimapped")
            if pct is not None:
                line = f"  {sname}: {pct}% uniquely mapped"
                if n and n_in:
                    line += f" ({n:,} / {n_in:,} reads)"
                if multi:
                    line += f", {multi}% multi-mapped"
                lines.append(line)
            else:
                lines.append(f"  {sname}: alignment stats not available")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Tool 4 — scRNA cluster markers                                       #
    # ------------------------------------------------------------------ #

    @tool
    def read_cluster_markers(cluster: str = "") -> str:
        """Read scRNA-seq marker genes for Leiden clusters.

        Use this when asked about which genes define a specific cluster,
        what cell types clusters might represent, or cluster identity.

        cluster: specific cluster ID (e.g. '0', '3'), or empty for all.
        """
        pr = pipeline_results or {}
        all_markers = pr.get("markers_top") or []
        if not all_markers:
            return "Cluster marker data not available (scRNA run required)."

        if cluster:
            subset = [m for m in all_markers if str(m["cluster"]) == str(cluster)]
            if not subset:
                available = sorted({str(m["cluster"]) for m in all_markers})
                return (
                    f"Cluster '{cluster}' not found. "
                    f"Available clusters: {available}"
                )
        else:
            subset = all_markers

        header = (
            f"Marker genes for cluster {cluster}"
            if cluster
            else "Marker genes per cluster"
        )
        lines = [header + ":"]
        current: str | None = None
        for m in subset:
            if str(m["cluster"]) != current:
                current = str(m["cluster"])
                lines.append(f"  Cluster {current}:")
            direction = "▲" if m["logfoldchange"] >= 0 else "▼"
            lines.append(
                f"    {direction} {m['gene']:<12} "
                f"log2FC={m['logfoldchange']:+.2f}  "
                f"padj={m['pval_adj']:.2e}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Tool 5 — run summary                                                 #
    # ------------------------------------------------------------------ #

    @tool
    def read_run_summary() -> str:
        """Read the full pipeline run summary.

        Use this for general questions about the experiment: what type of
        data was processed, how many samples/cells/clusters, conditions,
        what pipeline steps ran, overall QC status.
        """
        parts: list[str] = []
        if experiment_summary:
            parts.append(experiment_summary)

        pr = pipeline_results or {}
        data_type = pr.get("data_type", "bulk_rnaseq")

        if data_type == "scrna":
            if pr.get("n_cells") is not None:
                parts.append(f"Cells after QC + filtering: {pr['n_cells']:,}")
            if pr.get("n_genes") is not None:
                parts.append(f"Genes after filtering: {pr['n_genes']:,}")
            if pr.get("n_clusters") is not None:
                parts.append(f"Leiden clusters identified: {pr['n_clusters']}")
        else:
            conds = pr.get("conditions") or {}
            if conds:
                parts.append(
                    f"Comparison: {conds.get('treated','?')} vs "
                    f"{conds.get('reference','?')}"
                )
            if pr.get("n_samples_ok") is not None:
                parts.append(f"Samples completed: {pr['n_samples_ok']}")
            if pr.get("n_deg_significant") is not None:
                parts.append(f"Significant DEGs: {pr['n_deg_significant']}")
            n_inf = pr.get("inference_n_predicted")
            if n_inf is not None:
                parts.append(f"GNN PPI predictions: {n_inf}")

        ctx = pipeline_context or {}
        child_summary = ctx.get("child_summary") or ""
        if child_summary and child_summary not in "\n".join(parts):
            parts.append(f"\nPipeline summary:\n{child_summary}")

        return "\n".join(parts) if parts else "No run summary available."

    # ------------------------------------------------------------------ #

    return [
        retrieve_pathways,
        read_deg_table,
        read_alignment_stats,
        read_cluster_markers,
        read_run_summary,
    ]
