"""LangGraph node functions for the interactive RAG Q&A subgraph.

Graph shape:
    START → infer_data_type → collect_query → _route_after_query → END
                                                       ↓ (not quit)
                                                  rag_answer → collect_query

Nodes
-----
graph_node_infer_data_type
    Runs once at startup: LLM reads all pipeline artifacts, characterises
    the experiment in bullet points, and asks the user to confirm.

graph_node_collect_query
    Prompts the user for a free-text biology question.

graph_node_rag_answer
    Agentic: builds a ReAct agent with five data-source tools and lets
    the LLM decide which tools to call per question.  No keyword routing.

Router
------
_route_after_query(state) → "rag_answer" | END
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from scripts.rag.graph_state import RagChatState

# ---------------------------------------------------------------------------
# shared constants
# ---------------------------------------------------------------------------

_QUIT_SIGNALS = {"", "quit", "exit", "q", "done", "bye"}

# ---------------------------------------------------------------------------
# node 2 — agentic answer (ReAct loop over tools)
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = """\
You are a bioinformatics assistant answering questions about a pipeline run.
You have tools that fetch real data — always call at least one tool before
answering so your answer is grounded in the actual results.

Tool selection guide:
  retrieve_pathways    — pathway mechanisms, gene functions, enrichment
  read_deg_table       — which genes changed, fold changes, p-values
  read_alignment_stats — mapping rates, read counts, alignment quality
  read_cluster_markers — scRNA cluster identity, marker genes per cluster
  read_run_summary     — experiment type, conditions, sample/cell counts

Rules:
- Be concise: 3-6 sentences.
- Cite specific numbers from the tool output.
- For pathway questions, cite documents with [N].
- Do not speculate beyond what the tools return.
"""


def graph_node_rag_answer(state: RagChatState) -> dict:
    """Agentic answer: ReAct loop that calls tools until it can answer."""
    question = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        return {"messages": [AIMessage(content="(no question received)")]}

    from scripts.rag.answerer import _make_llm
    from scripts.rag.tools import build_rag_tools

    llm = _make_llm()
    if llm is None:
        return {
            "messages": [
                AIMessage(
                    content="No LLM configured. Set OLLAMA_MODEL or OPENAI_API_KEY."
                )
            ]
        }

    pr = state.get("pipeline_results") or {}
    arts = state.get("run_artifacts") or {}
    ctx = state.get("pipeline_context") or {}
    exp = state.get("experiment_summary") or ""

    tools = build_rag_tools(pr, arts, ctx, exp)

    system = _AGENT_SYSTEM
    if exp:
        system += f"\nExperiment context:\n{exp}"

    try:
        from langgraph.prebuilt import create_react_agent

        try:
            agent = create_react_agent(llm, tools, prompt=system)
        except TypeError:
            # older langgraph: prompt= not yet available
            from langchain_core.messages import SystemMessage

            def _mod(msgs: list) -> list:
                return [SystemMessage(content=system)] + list(msgs)

            agent = create_react_agent(llm, tools, messages_modifier=_mod)

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 12},
        )
        answer_text = result["messages"][-1].content
    except Exception as exc:
        answer_text = f"[Agent error: {type(exc).__name__}: {exc}]"

    print(f"\nAnswer:\n{answer_text}\n")
    return {"messages": [AIMessage(content=answer_text)]}


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
    "Based on this data, characterise the experiment in 4–6 bullet points "
    "covering:\n"
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

    Skipped if state['data_type_confirmed'] is already True.
    """
    if state.get("data_type_confirmed"):
        return {}

    arts = state.get("run_artifacts") or {}
    pr = state.get("pipeline_results") or {}
    data_type = pr.get("data_type", "bulk_rnaseq")

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
        from langchain_core.messages import SystemMessage

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
            # Deterministic fallback
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
                n_ok = pr.get("n_samples_ok")
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

    print()
    return {
        "experiment_summary": experiment_summary,
        "data_type_confirmed": confirmed,
    }


# ---------------------------------------------------------------------------
# node 1 — collect the user's question
# ---------------------------------------------------------------------------


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
# conditional router
# ---------------------------------------------------------------------------


def _route_after_query(state: RagChatState) -> str:
    """If should_quit is True, go to END; otherwise ask + answer."""
    return "__end__" if state.get("should_quit") else "rag_answer"
