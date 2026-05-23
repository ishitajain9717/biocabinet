"""Render the bulk RNA-seq and scRNA-seq pipeline graphs as PNGs + mermaid text.

Run from project root:
    python3 -m scripts.render_graphs

Outputs:
    docs/bulk_graph.png
    docs/scrna_graph.png
    docs/bulk_graph.mmd
    docs/scrna_graph.mmd
"""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from scripts.bulk_rnaseq.graph  import build_graph as build_bulk_graph
from scripts.enrichment.graph   import build_graph as build_enrichment_graph
from scripts.orchestrator       import build_graph as build_orchestrator_graph
from scripts.scrna.graph        import build_graph as build_scrna_graph
from scripts.spatial.graph      import build_graph as build_spatial_graph


def _render(workflow, label: str, out_dir: Path) -> None:
    compiled = workflow.compile()
    g = compiled.get_graph()

    mmd_path = out_dir / f"{label}_graph.mmd"
    png_path = out_dir / f"{label}_graph.png"

    mmd_path.write_text(g.draw_mermaid())
    print(f"  wrote {mmd_path}")

    try:
        png_bytes = g.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"  wrote {png_path}")
    except Exception as exc:
        print(f"  (PNG render skipped: {type(exc).__name__}: {exc})")
        print(f"  → render the .mmd file at https://mermaid.live instead")


def main() -> None:
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)

    print("Rendering bulk RNA-seq graph...")
    _render(build_bulk_graph(), "bulk", out_dir)

    print("\nRendering scRNA-seq graph...")
    _render(build_scrna_graph(), "scrna", out_dir)

    print("\nRendering spatial stub graph...")
    _render(build_spatial_graph(), "spatial", out_dir)

    print("\nRendering enrichment (GNN-PPI) graph...")
    _render(build_enrichment_graph(), "enrichment", out_dir)

    print("\nRendering orchestrator graph...")
    with SqliteSaver.from_conn_string(":memory:") as mem:
        _render(build_orchestrator_graph(mem, "render"), "orchestrator", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
