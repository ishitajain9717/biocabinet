"""Build, compile, and run the scRNA-seq pipeline as a LangGraph.

Flow (trajectory enabled)
--------------------------
    collect_config → load_data → qc → filter → normalize → pca → cluster
        → markers → [trajectory] → [palantir ← INTERRUPT] → summarize → END
                 ↓ (any failure)
              error_node → END

When ``run_trajectory=False`` the graph goes directly markers → summarize,
skipping both trajectory and palantir.

Palantir interrupt
------------------
The graph is compiled with ``interrupt_before=["palantir"]``.  After PAGA
runs the pipeline pauses, prints the cluster connectivity and plot paths,
and asks the biologist which Leiden cluster is the biological root.  The
answer is written back into state via ``graph.update_state``, then the run
resumes automatically.

Run a fresh pipeline:
    python3 -m scripts.scrna.graph

Resume a previous run:
    python3 -m scripts.scrna.graph --thread-id scrna_run_20260428_094200
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from scripts.scrna.graph_nodes import (
    _route_after_benchmark,
    _route_after_markers,
    graph_node_benchmark,
    graph_node_cluster,
    graph_node_collect_config,
    graph_node_error,
    graph_node_filter,
    graph_node_load_data,
    graph_node_markers,
    graph_node_normalize,
    graph_node_palantir,
    graph_node_pca,
    graph_node_qc,
    graph_node_summarize,
    graph_node_trajectory,
)
from scripts.scrna.graph_state import ScrnaState


def _route_or_continue(next_step: str):
    """Build a conditional-edge router: error_node if state.error, else next_step."""

    def router(state: ScrnaState) -> str:
        return "error_node" if state.get("error") else next_step

    return router


def build_graph() -> StateGraph:
    """Construct the scRNA-seq StateGraph (not yet compiled)."""
    workflow = StateGraph(ScrnaState)

    workflow.add_node("collect_config", graph_node_collect_config)
    workflow.add_node("load_data", graph_node_load_data)
    workflow.add_node("qc", graph_node_qc)
    workflow.add_node("filter", graph_node_filter)
    workflow.add_node("normalize", graph_node_normalize)
    workflow.add_node("pca", graph_node_pca)
    workflow.add_node("cluster", graph_node_cluster)
    workflow.add_node("markers", graph_node_markers)
    workflow.add_node("benchmark", graph_node_benchmark)
    workflow.add_node("trajectory", graph_node_trajectory)
    workflow.add_node("palantir", graph_node_palantir)
    workflow.add_node("summarize", graph_node_summarize)
    workflow.add_node("error_node", graph_node_error)

    workflow.set_entry_point("collect_config")
    workflow.add_edge("collect_config", "load_data")

    for prev, nxt in [
        ("load_data", "qc"),
        ("qc", "filter"),
        ("filter", "normalize"),
        ("normalize", "pca"),
        ("pca", "cluster"),
        ("cluster", "markers"),
    ]:
        workflow.add_conditional_edges(
            prev,
            _route_or_continue(nxt),
            {nxt: nxt, "error_node": "error_node"},
        )

    # After markers: benchmark (opt) → trajectory (opt) → summarize.
    workflow.add_conditional_edges(
        "markers",
        _route_after_markers,
        {
            "benchmark": "benchmark",
            "trajectory": "trajectory",
            "summarize": "summarize",
            "error_node": "error_node",
        },
    )

    # After benchmark: go to trajectory if requested, else summarize.
    # Benchmark is non-fatal so no error_node edge needed.
    workflow.add_conditional_edges(
        "benchmark",
        _route_after_benchmark,
        {"trajectory": "trajectory", "summarize": "summarize"},
    )

    # Trajectory is non-fatal → always advance to palantir (which self-skips
    # if trajectory_path is None).  Palantir is interrupted here by the
    # interrupt_before=["palantir"] compile option.
    workflow.add_edge("trajectory", "palantir")
    workflow.add_edge("palantir", "summarize")

    workflow.add_edge("summarize", END)
    workflow.add_edge("error_node", END)

    return workflow


# ---------------------------------------------------------------------------
# Palantir interrupt helpers
# ---------------------------------------------------------------------------


def _prompt_root_cluster(state: dict) -> str:
    """Display PAGA results and ask the biologist for the root cluster.

    Called when the graph is paused just before the palantir node.
    """
    n_clusters = state.get("n_clusters", "?")
    traj_path = state.get("trajectory_path")

    print("\n" + "═" * 60)
    print("  Palantir requires a root cluster — human input needed")
    print("═" * 60)

    if traj_path and Path(traj_path).exists():
        try:
            data = json.loads(Path(traj_path).read_text())
            print(f"\n  Clusters : {data['n_clusters']}")
            print("  PAGA edges (top 5 by weight):")
            for e in sorted(data["edges"], key=lambda e: -e["weight"])[:5]:
                bar = "█" * int(e["weight"] * 20)
                print(
                    f"    {e['source']:>3} — {e['target']:<3}  {bar}  {e['weight']:.3f}"
                )
        except Exception:
            pass

        traj_dir = Path(traj_path).parent
        print(f"\n  PAGA plot : {traj_dir}/paga_connectivity.png")
        print(f"  UMAP      : {traj_dir}/umap_paga_umap.png")

    markers_path = state.get("markers_path")
    if markers_path:
        print(f"  Markers   : {markers_path}")

    print()
    cluster_range = f"0–{n_clusters - 1}" if isinstance(n_clusters, int) else "?"
    while True:
        raw = input(f"  Enter root cluster ID [{cluster_range}]: ").strip()
        if raw:
            return raw
        print("  Please enter a cluster ID.")


def _run_with_interrupt(
    compiled,
    initial_state: dict,
    config: dict,
) -> tuple[str, str | None, dict | None]:
    """Invoke the compiled scRNA graph, handling the Palantir interrupt.

    Returns ``(summary, error, final_state)``.

    Flow
    ----
    1. First invoke runs until just before ``palantir`` (or to completion if
       trajectory was skipped).
    2. If paused at ``palantir`` AND PAGA succeeded, prompt the biologist for
       the root cluster, update state, then resume.
    3. If paused but PAGA failed (trajectory_path is None), just resume —
       graph_node_palantir will self-skip with a warning.
    """
    try:
        compiled.invoke(initial_state, config=config)
    except Exception as exc:
        return ("", f"{type(exc).__name__}: {exc}", None)

    snapshot = compiled.get_state(config)
    interrupted = "palantir" in list(snapshot.next or [])

    if interrupted:
        state_vals = snapshot.values or {}
        if state_vals.get("trajectory_path"):
            root_cluster = _prompt_root_cluster(state_vals)
        else:
            # PAGA didn't run — palantir will self-skip; no user input needed.
            print(
                "  [palantir] PAGA trajectory not available — skipping Palantir.",
                flush=True,
            )
            root_cluster = None

        compiled.update_state(config, {"root_cluster": root_cluster})

        try:
            compiled.invoke(None, config=config)
        except Exception as exc:
            return ("", f"{type(exc).__name__}: {exc}", None)

        snapshot = compiled.get_state(config)

    final = snapshot.values or {}
    summary = ""
    if final.get("messages"):
        summary = final["messages"][-1].content
    error = final.get("error")
    return (summary, error, final)


# ---------------------------------------------------------------------------
# Public entry point used by both standalone and orchestrator
# ---------------------------------------------------------------------------


def run_scrna_with_interrupt(
    memory,
    thread_id: str,
) -> tuple[str, str | None, dict | None]:
    """Build, compile (with interrupt_before=['palantir']), and run the graph.

    Returns ``(summary, error, final_state)`` — same shape as
    ``_run_child(..., return_state=True)`` so the orchestrator can consume it
    without special-casing.
    """
    workflow = build_graph()
    compiled = workflow.compile(
        checkpointer=memory,
        interrupt_before=["palantir"],
    )
    config = {"configurable": {"thread_id": thread_id}}
    return _run_with_interrupt(compiled, {}, config)


# ---------------------------------------------------------------------------
# Standalone runner (python3 -m scripts.scrna.graph)
# ---------------------------------------------------------------------------


def run_pipeline(thread_id: str | None = None) -> dict:
    """Compile and invoke the scRNA-seq graph. Returns the final state."""
    if thread_id is None:
        thread_id = "scrna_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    memory = MemorySaver()

    print(f"\n=== Starting scRNA-seq pipeline (thread_id={thread_id}) ===\n")
    summary, error, final = run_scrna_with_interrupt(memory, thread_id)

    if summary:
        print("\n=== Run summary ===")
        print(summary)
    if error:
        print(f"\n[ERROR] {error}")

    return final or {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default=None)
    args = parser.parse_args()
    run_pipeline(thread_id=args.thread_id)
