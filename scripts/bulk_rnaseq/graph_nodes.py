"""LangGraph node functions.

Each function takes the full PipelineState and returns a partial dict
of state updates. LangGraph merges that dict back into the state.

The LLM summarizer node dispatches between providers via LLM_PROVIDER:
  - LLM_PROVIDER=ollama (default) → local Ollama, model from OLLAMA_MODEL
  - LLM_PROVIDER=openai           → OpenAI, model from OPENAI_MODEL,
                                    needs OPENAI_API_KEY
Any failure (no provider, network down, no quota) falls back to a
deterministic text summary so the pipeline always completes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scripts.bulk_rnaseq.config import collect_config_from_user
from scripts.bulk_rnaseq.graph_state import PipelineState
from scripts.bulk_rnaseq.nodes import node_deg, node_normalize
from scripts.bulk_rnaseq.run_preprocessing import _run_sample
from scripts.common.node_result import NodeResult

# ---------- node 1: ask user for config ----------

_QC_SYSTEM_PROMPT = (
    "You are a bioinformatics quality-control assistant. "
    "You will be given FastQC metrics for one RNA-seq sample. "
    "Decide whether the sample should proceed to alignment.\n\n"
    "Respond with EXACTLY this format (no extra text):\n"
    "DECISION: PASS|WARN|FAIL\n"
    "REASON: <one sentence>\n\n"
    "Guidelines:\n"
    "- FAIL if: mean_quality < 20, OR >50% adapter content, OR "
    "Per base sequence quality is 'fail'\n"
    "- WARN if: mean_quality 20-27, OR >20% adapter content, OR "
    "multiple modules are 'warn'\n"
    "- PASS otherwise\n"
    "- Per sequence GC content 'warn' alone is not a reason to fail\n"
    "- High duplication is expected for RNA-seq; do not fail on it alone"
)


def _llm_fastqc_gate(sample_name: str, fastqc_metrics: dict) -> tuple[str, str]:
    """Ask the LLM to decide PASS / WARN / FAIL for a FastQC report.

    Returns (decision, reason) where decision is one of
    'PASS', 'WARN', 'FAIL', or 'UNKNOWN' (if parsing fails).
    Falls back to rule-based decision if no LLM is configured.
    """
    modules = fastqc_metrics.get("module_statuses") or {}
    mean_q = fastqc_metrics.get("mean_quality")
    pct_dup = fastqc_metrics.get("pct_duplicates")
    pct_ada = fastqc_metrics.get("pct_adapter")
    pct_gc = fastqc_metrics.get("pct_gc")
    n_seq = fastqc_metrics.get("total_sequences")

    # Format a compact QC summary for the LLM
    module_lines = "\n".join(
        f"  {mod}: {status}" for mod, status in modules.items() if mod != "FastQC"
    )
    prompt = (
        f"Sample: {sample_name}\n"
        f"Total sequences : {n_seq}\n"
        f"Mean quality    : {mean_q}\n"
        f"% GC content    : {pct_gc}\n"
        f"% duplicates    : {pct_dup}\n"
        f"% adapter       : {pct_ada}\n\n"
        f"Module statuses:\n{module_lines}\n\n"
        "Should this sample proceed to alignment?"
    )

    llm = _make_llm()
    if llm is not None:
        try:
            resp = llm.invoke(
                [
                    SystemMessage(content=_QC_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            text = resp.content.strip()
            # Parse DECISION: and REASON: lines
            dec_match = re.search(r"DECISION:\s*(PASS|WARN|FAIL)", text, re.IGNORECASE)
            rea_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
            if dec_match:
                decision = dec_match.group(1).upper()
                reason = rea_match.group(1).strip() if rea_match else text
                return decision, reason
        except Exception:
            pass  # fall through to rule-based

    # Rule-based fallback (no LLM)
    fails = [m for m, s in modules.items() if s == "fail"]
    warns = [m for m, s in modules.items() if s == "warn"]

    if (mean_q is not None and mean_q < 20) or (pct_ada is not None and pct_ada > 50):
        reason = f"mean quality={mean_q}, adapter={pct_ada}% — hard thresholds exceeded"
        return "FAIL", reason
    if "Per base sequence quality" in fails:
        return "FAIL", "Per base sequence quality failed"
    if (mean_q is not None and mean_q < 27) or len(warns) >= 3:
        return "WARN", f"{len(warns)} modules warned; mean quality={mean_q}"
    return "PASS", f"mean quality={mean_q}, adapter={pct_ada}%, {len(fails)} fails"


_TRIM_SYSTEM_PROMPT = (
    "You are a bioinformatics assistant generating Trimmomatic step arguments "
    "for an RNA-seq sample based on its FastQC report.\n\n"
    "Respond with EXACTLY this format (no extra text):\n"
    "STEPS: <space-separated Trimmomatic steps, or NONE if no trimming needed>\n"
    "REASON: <one sentence explaining the choice>\n\n"
    "Available Trimmomatic steps:\n"
    "  ILLUMINACLIP:<fasta>:<seed_mismatch>:<palindrome_clip>:<simple_clip>\n"
    "    Use TruSeq3-PE.fa or TruSeq3-SE.fa (included with Trimmomatic).\n"
    "    Add only when Adapter Content module is warn/fail or pct_adapter > 5%.\n"
    "  LEADING:<quality>      Remove low-quality bases from start (use 20).\n"
    "  TRAILING:<quality>     Remove low-quality bases from end (use 20).\n"
    "  SLIDINGWINDOW:<w>:<q>  Cut when window quality drops below threshold.\n"
    "    Typical: SLIDINGWINDOW:4:15 for mild issues, SLIDINGWINDOW:4:20 for severe.\n"
    "  MINLEN:<length>        Drop reads shorter than this after trimming (36).\n\n"
    "Rules:\n"
    "- If adapter_pct > 5% or Adapter Content is warn/fail: include ILLUMINACLIP\n"
    "- If Per base sequence quality is warn: add LEADING:20 TRAILING:20 "
    "SLIDINGWINDOW:4:15\n"
    "- If Per base sequence quality is fail: use SLIDINGWINDOW:4:20 (aggressive)\n"
    "- If all modules pass and adapter < 2%: respond with STEPS: NONE\n"
    "- Always end with MINLEN:36 when any trimming steps are included\n"
    "- Do NOT invent file paths; use TruSeq3-PE.fa or TruSeq3-SE.fa as-is"
)

_TRIM_STEP_SAFE_RE = re.compile(
    r"^(ILLUMINACLIP:[^\s:]+:\d+:\d+:\d+"
    r"|LEADING:\d+"
    r"|TRAILING:\d+"
    r"|SLIDINGWINDOW:\d+:\d+"
    r"|MINLEN:\d+)$"
)


def _llm_trim_gate(
    sample_name: str,
    fastqc_metrics: dict,
    paired_end: bool = True,
) -> tuple[list[str], str]:
    """Ask the LLM to generate the Trimmomatic steps for this sample.

    Returns (trim_steps, reason) where:
    - trim_steps is a list of validated Trimmomatic step tokens, e.g.
      ["ILLUMINACLIP:TruSeq3-PE.fa:2:30:10", "LEADING:20", "TRAILING:20",
       "SLIDINGWINDOW:4:15", "MINLEN:36"]
    - trim_steps == [] means the LLM decided no trimming is needed

    Falls back to rule-based step generation if no LLM is configured.
    """
    modules = fastqc_metrics.get("module_statuses") or {}
    mean_q = fastqc_metrics.get("mean_quality")
    pct_ada = fastqc_metrics.get("pct_adapter")

    adapter_fa = "TruSeq3-PE.fa" if paired_end else "TruSeq3-SE.fa"
    module_lines = "\n".join(
        f"  {mod}: {status}" for mod, status in modules.items() if mod != "FastQC"
    )
    prompt = (
        f"Sample       : {sample_name}\n"
        f"Library type : {'paired-end' if paired_end else 'single-end'}\n"
        f"Mean quality : {mean_q}\n"
        f"% adapter    : {pct_ada}\n\n"
        f"Module statuses:\n{module_lines}\n\n"
        "Generate the Trimmomatic step arguments for this sample."
    )

    llm = _make_llm()
    if llm is not None:
        try:
            print(
                f"  [Trim gate] Asking LLM for trim steps "
                f"(sample={sample_name})...",
                flush=True,
            )
            resp = llm.invoke(
                [
                    SystemMessage(content=_TRIM_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            text = resp.content.strip()
            steps_match = re.search(r"STEPS:\s*(.+)", text, re.IGNORECASE)
            rea_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
            reason = rea_match.group(1).strip() if rea_match else text

            if steps_match:
                raw_steps = steps_match.group(1).strip()
                if raw_steps.upper() == "NONE":
                    return [], reason

                # Validate each token against the safe regex — reject anything
                # the LLM hallucinated that doesn't look like a real step
                validated: list[str] = []
                rejected: list[str] = []
                for tok in raw_steps.split():
                    if _TRIM_STEP_SAFE_RE.match(tok):
                        validated.append(tok)
                    else:
                        rejected.append(tok)

                if rejected:
                    print(
                        f"  [Trim gate] Rejected unsafe tokens: {rejected}"
                        " — falling back to rule-based",
                        flush=True,
                    )
                    # Don't use partial results; fall through to rule-based
                elif validated:
                    # Full clean output from LLM — apply sanity checks
                    has_clip = any(s.startswith("ILLUMINACLIP") for s in validated)
                    adapter_clean = (pct_ada is None or pct_ada < 5) and modules.get(
                        "Adapter Content", "pass"
                    ) == "pass"
                    if has_clip and adapter_clean:
                        validated = [
                            s for s in validated if not s.startswith("ILLUMINACLIP")
                        ]
                        print(
                            "  [Trim gate] Removed ILLUMINACLIP "
                            f"(adapter clean at {pct_ada}%)",
                            flush=True,
                        )
                    # If only MINLEN is left after stripping, no real trimming
                    real_steps = [s for s in validated if not s.startswith("MINLEN")]
                    if not real_steps:
                        return [], f"no trimming needed ({reason})"
                    return validated, reason
                else:
                    # LLM said STEPS: NONE and we matched it above
                    return [], reason
        except Exception:
            pass  # fall through to rule-based

    # ── Rule-based fallback ──────────────────────────────────────────────────
    adapter_status = modules.get("Adapter Content", "pass")
    quality_status = modules.get("Per base sequence quality", "pass")

    steps: list[str] = []
    reasons: list[str] = []

    if adapter_status in ("warn", "fail") or (pct_ada and pct_ada > 5):
        steps.append(f"ILLUMINACLIP:{adapter_fa}:2:30:10")
        reasons.append(f"adapter={pct_ada}%")

    if quality_status == "fail":
        steps += ["LEADING:20", "TRAILING:20", "SLIDINGWINDOW:4:20"]
        reasons.append("quality=fail (aggressive window)")
    elif quality_status == "warn":
        steps += ["LEADING:20", "TRAILING:20", "SLIDINGWINDOW:4:15"]
        reasons.append("quality=warn")

    if steps:
        steps.append("MINLEN:36")
        return steps, "; ".join(reasons)

    return [], f"adapter={pct_ada}% (<2%) and all quality modules pass"


# ---------- node 1: collect config + seed state ----------


def graph_node_collect_config(state: PipelineState) -> dict:
    """Prompt the user for paths/options and seed the state.

    Returns initial empty values for every field the rest of the graph
    will read, so downstream nodes never hit a KeyError.
    """
    cfg = collect_config_from_user()
    return {
        "config": cfg,
        "node_history": [],
        "count_results": [],
        "failed_samples": [],
        "qc_decisions": {},
        "norm_outputs": {},
        "deg_full_path": None,
        "deg_sig_path": None,
        "deg_pairs_path": None,
        "n_deg_significant": None,
        "error": None,
    }


# ---------- node 2: run all samples through fastqc/trim/align/featurecounts ----------


def graph_node_run_samples(state: PipelineState) -> dict:
    """Loop over every sample, run FastQC → LLM gate → trim → align → counts."""
    cfg = state["config"]
    history: list[NodeResult] = list(state["node_history"])
    count_results: list[tuple[str, str]] = []
    failed: list[str] = []
    qc_decisions: dict[str, dict] = dict(state.get("qc_decisions") or {})

    for sample in cfg.samples:
        print(f"\n--- Sample: {sample.name} ---")
        sample_history, counts_txt, qc_dec = _run_sample(
            cfg,
            sample,
            llm_fastqc_gate_fn=_llm_fastqc_gate,
            llm_trim_gate_fn=_llm_trim_gate,
        )
        history.extend(sample_history)

        if qc_dec is not None:
            qc_decisions[sample.name] = qc_dec

        for r in sample_history:
            status = "OK  " if r.ok else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")

        if counts_txt is None:
            failed.append(sample.name)
            print(f"  => stopped early for {sample.name}")
        else:
            # Convert Path -> str so the SQLite checkpointer can JSON-serialise it.
            count_results.append((sample.name, str(counts_txt)))

    return {
        "node_history": history,
        "count_results": count_results,
        "failed_samples": failed,
        "qc_decisions": qc_decisions,
    }


# ---------- node 3: merge counts and write TPM/FPKM/RPKM ----------


def graph_node_normalize(state: PipelineState) -> dict:
    """Run node_normalize on whatever samples produced counts.

    If no samples succeeded, sets state['error'] so the graph can route
    away from the summarizer to an error handler.
    """
    cfg = state["config"]

    if not state["count_results"]:
        return {"error": "No successful samples — nothing to normalize."}

    # Convert (str, str) tuples back to (str, Path) for node_normalize.
    count_results = [(name, Path(p)) for name, p in state["count_results"]]
    result = node_normalize(cfg, count_results)

    history = list(state["node_history"]) + [result]
    return {
        "node_history": history,
        "norm_outputs": result.outputs if result.ok else {},
        "error": None if result.ok else result.message,
    }


# ---------- node 4: differential expression (PyDESeq2) ----------


def graph_node_deg(state: PipelineState) -> dict:
    """Run DEG analysis on the raw counts matrix produced by `normalize`.

    No-ops (returns ok=True with a 'skipped' message) when the user opted
    out at config time. On success, populates state['deg_pairs_path'] which
    the orchestrator's enrichment auto-chain reads as candidate pairs.
    """
    cfg = state["config"]
    norm_outputs = state.get("norm_outputs") or {}
    raw_counts = norm_outputs.get("counts_raw")

    # Sanity: we need the raw counts matrix even when DEG is enabled. If it's
    # missing we skip rather than crash, so the rest of the pipeline still
    # finishes cleanly.
    if not raw_counts:
        from scripts.common.node_result import NodeResult

        result = NodeResult(
            name="deg",
            ok=True,
            message="skipped: no counts_raw produced upstream (nothing to test)",
            metrics={"skipped": True},
        )
        return {"node_history": list(state["node_history"]) + [result]}

    result = node_deg(cfg, Path(raw_counts))
    history = list(state["node_history"]) + [result]

    if not result.ok:
        return {
            "node_history": history,
            # node_deg failures are loud but should not halt the pipeline at
            # the LangGraph level; the summarize node will report it.
            "deg_full_path": None,
            "deg_sig_path": None,
            "deg_pairs_path": None,
            "n_deg_significant": None,
        }

    out = result.outputs or {}
    m = result.metrics or {}
    return {
        "node_history": history,
        "deg_full_path": out.get("deseq2_full"),
        "deg_sig_path": out.get("deseq2_significant"),
        "deg_pairs_path": out.get("deg_pairs"),
        "n_deg_significant": m.get("n_significant"),
    }


# ---------- node 5: summarize the whole run (LLM, with fallback) ----------


def _deterministic_summary(state: PipelineState) -> str:
    """Plain-Python text summary used when no OPENAI_API_KEY is set."""
    history = state["node_history"]
    n_ok = sum(1 for r in history if r.ok)
    n_fail = sum(1 for r in history if not r.ok)
    failed = state["failed_samples"]
    norms = list(state["norm_outputs"].keys())
    qc_decisions = state.get("qc_decisions") or {}

    lines = [
        "RNA-seq preprocessing run complete.",
        f"  Nodes succeeded : {n_ok}",
        f"  Nodes failed    : {n_fail}",
        f"  Failed samples  : {failed if failed else 'none'}",
        f"  Normalizations  : {norms if norms else 'none'}",
    ]

    # Per-sample alignment stats from node_history (align nodes)
    align_stats: dict[str, dict] = {}
    for r in history:
        if r.name == "align" and r.metrics:
            # Try to recover sample name from output bam path
            bam = r.outputs.get("bam", "")
            # bam path is <out_dir>/03_bam/<sample>/<sample>_Aligned...
            parts = Path(bam).parts if bam else []
            sname_guess = parts[-2] if len(parts) >= 2 else "unknown"
            align_stats[sname_guess] = r.metrics

    if align_stats:
        lines.append("\nAlignment stats (STAR):")
        for sname, m in align_stats.items():
            pct_map = m.get("pct_uniquely_mapped")
            pct_multi = m.get("pct_multi_mapped")
            pct_short = m.get("pct_unmapped_tooshort")
            n_input = m.get("n_input_reads")
            mismatch = m.get("mismatch_rate_pct")
            flag = " [LOW MAPPING]" if m.get("mapping_rate_ok") is False else ""
            stat_parts = []
            if n_input:
                stat_parts.append(f"input={n_input:,}")
            stat_parts.append(f"unique={pct_map}%{flag}")
            if pct_multi is not None:
                stat_parts.append(f"multi={pct_multi}%")
            if pct_short is not None:
                stat_parts.append(f"too_short={pct_short}%")
            if mismatch is not None:
                stat_parts.append(f"mismatch={mismatch}%")
            lines.append(f"  {sname:20s}  " + "  ".join(stat_parts))

    if qc_decisions:
        lines.append("\nQC + trim gate decisions (LLM):")
        for sname, dec in qc_decisions.items():
            verdict = dec.get("decision", "?")
            reason = dec.get("reason", "")
            steps = dec.get("trim_steps", [])
            trim_reason = dec.get("trim_reason", "")
            steps_str = " ".join(steps) if steps else "SKIP-TRIM"
            lines.append(f"  {sname:20s}  [QC:{verdict}]  {reason}")
            lines.append(f"  {'':20s}  [TRIM: {steps_str}]")
            if trim_reason:
                lines.append(f"  {'':20s}         {trim_reason}")

    if n_fail:
        lines.append("\nFailures:")
        for r in history:
            if not r.ok:
                lines.append(f"  - {r.name}: {r.message}")

    return "\n".join(lines)


def _make_llm():
    """Return a LangChain chat model based on LLM_PROVIDER, or None to fall back.

    LLM_PROVIDER=ollama (default) → ChatOllama, model = OLLAMA_MODEL or qwen2.5:3b
    LLM_PROVIDER=openai           → ChatOpenAI, model = OPENAI_MODEL or gpt-4o-mini
                                     (requires OPENAI_API_KEY)
    Anything else / unset → None  → use deterministic summary
    """
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()

    if provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        # Check server is up before attempting a connection that would hang.
        try:
            import urllib.request

            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            print(
                "[summarize] Ollama server not reachable — falling back to "
                "deterministic summary.  Start Ollama with: ollama serve",
                flush=True,
            )
            return None
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=ollama_model,
            temperature=0,
            keep_alive=-1,
            client_kwargs={"timeout": 60},
        )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=60,
        )

    return None


def graph_node_summarize(state: PipelineState) -> dict:
    """Use an LLM to write a friendly summary of the run, then append RAG
    pathway context for any significant DEGs that were found.

    Falls back to a deterministic text summary if no LLM is configured.
    The RAG augmentation is always attempted after the LLM step but never
    blocks pipeline completion (errors are silently swallowed).
    """
    history_lines = "\n".join(
        f"  - {r.name}: {'OK' if r.ok else 'FAIL'} — {r.message}"
        for r in state["node_history"]
    )
    user_prompt = f"""Summarize this RNA-seq preprocessing run.

Node results:
{history_lines}

Failed samples: {state['failed_samples'] or 'none'}
Normalizations produced: {list(state['norm_outputs'].keys()) or 'none'}

Write 3-5 sentences. Be specific: how many samples succeeded, what tools ran,
what output files exist, and which (if any) failures need attention."""

    llm = _make_llm()
    if llm is None:
        base = _deterministic_summary(state)
    else:
        print("[summarize] Generating summary with LLM (may take ~10 s)...", flush=True)
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content="You are a careful bioinformatics assistant."
                    ),
                    HumanMessage(content=user_prompt),
                ]
            )
            base = response.content
        except Exception as exc:
            base = (
                f"[LLM summary failed: {type(exc).__name__}: {exc}]\n\n"
                + _deterministic_summary(state)
            )

    # Phase 4a: append RAG pathway context (non-blocking)
    try:
        from scripts.rag.augment import rag_augment_bulk

        rag_extra = rag_augment_bulk(dict(state))
    except Exception:
        rag_extra = ""

    full = base + rag_extra if rag_extra else base
    return {"messages": [AIMessage(content=full)]}
