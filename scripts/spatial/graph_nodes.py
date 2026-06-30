"""LLM adjudication helpers for the spatial pipeline.

Same pattern as the bulk FastQC / trim gates: deterministic code computes an
evidence packet (in nodes.py), and the LLM here makes the judgment call over
that evidence, with a rule-based fallback and hard-metric guardrails so a
hallucinating small model can never drive a wrong action unchecked.
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# LLM factory (local, mirrors the bulk/RAG pattern)
# ---------------------------------------------------------------------------


def _make_llm():
    """Return a LangChain chat model, or None to fall back to rules."""
    provider = (os.getenv("LLM_PROVIDER") or "ollama").lower()

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        try:
            import urllib.request

            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            return None
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
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


# ---------------------------------------------------------------------------
# FOV-bias adjudication
# ---------------------------------------------------------------------------

_FOV_SYSTEM_PROMPT = (
    "You are a spatial-omics QC assistant. A clustering was run on imaging-"
    "based spatial data (MERSCOPE). Each cell belongs to a field-of-view "
    "(FOV) imaging tile. You must decide whether suspicious cluster pairs "
    "reflect FOV batch effect (one real cell type split by tile) or genuine "
    "biology (distinct types, or a rare spatially-restricted type).\n\n"
    "Key reasoning:\n"
    "- BATCH EFFECT signs: clusters are transcriptomically near-identical "
    "(high centroid similarity / profile correlation), are dominated by "
    "DIFFERENT single FOVs (fov_disjoint), differ mainly by a uniform "
    "magnitude shift (de_type=magnitude_shift), and global iLISI is low "
    "(neighbors segregate by FOV).\n"
    "- BIOLOGY signs: the separating genes are real markers (de_type="
    "markers), iLISI is healthy, or only a few isolated cells segregate "
    "(a rare local cell type).\n\n"
    "Respond EXACTLY in this format:\n"
    "DECISION: BATCH_EFFECT | BIOLOGY | UNCERTAIN\n"
    "ACTION: correct_and_recluster | keep | flag_for_user\n"
    "REASON: <one sentence>"
)


def _format_evidence(evidence: dict) -> str:
    n_fov = evidence.get("n_fov")
    lines = [
        f"n_clusters={evidence.get('n_clusters')}  n_fov={n_fov}",
        f"median_iLISI={evidence.get('median_ilisi'):.2f} "
        f"(1=neighbors all one FOV, {n_fov}=perfectly mixed)",
        f"frac_cells_low_iLISI={evidence.get('frac_low_ilisi'):.2f}",
        f"cluster_vs_FOV_association(CramersV)={evidence.get('cramers_v'):.2f}"
        " (0=independent, 1=cluster fully determined by FOV)",
        f"frac_cells_in_FOV_pure_clusters="
        f"{evidence.get('frac_cells_fov_pure'):.2f}",
        f"n_FOV_pure_clusters={evidence.get('n_fov_pure_clusters')}"
        f"/{evidence.get('n_clusters')}",
    ]
    pure = evidence.get("fov_pure_clusters") or []
    if pure:
        lines.append("FOV-pure clusters (>=70% one FOV):")
        for c in pure:
            lines.append(
                f"  cluster {c['cluster']}: {c['dominant_fov']}"
                f"@{c['dominant_frac']:.0%} (n={c['n_cells']})"
            )
    pairs = evidence.get("suspicious_pairs") or []
    if pairs:
        lines.append("Near-identical cluster pairs:")
        for p in pairs:
            lines.append(
                f"  {p['cluster_a']}+{p['cluster_b']}: "
                f"sim={p['centroid_similarity']:.2f} "
                f"corr={p['profile_corr']:.2f} "
                f"fov_disjoint={p['fov_disjoint']} "
                f"({p['a_dominant_fov']}@{p['a_dominant_frac']:.0%} vs "
                f"{p['b_dominant_fov']}@{p['b_dominant_frac']:.0%}) "
                f"de_type={p['de_type']}"
            )
    return "\n".join(lines)


def _low_ilisi(evidence: dict) -> bool:
    n_fov = evidence.get("n_fov") or 1
    return evidence.get("median_ilisi", n_fov) < max(2.0, 0.5 * n_fov)


def _batch_signal(evidence: dict) -> bool:
    """Strong, hard-metric batch-effect fingerprint: clustering is largely
    explained by FOV, neighbors segregate by FOV, and most cells live in
    FOV-pure clusters."""
    return (
        evidence.get("cramers_v", 0.0) >= 0.45
        and _low_ilisi(evidence)
        and evidence.get("frac_cells_fov_pure", 0.0) >= 0.40
    )


def _no_signal(evidence: dict) -> bool:
    """Clean: clusters independent of FOV and neighbors mix well."""
    return (
        evidence.get("cramers_v", 0.0) < 0.25
        and not _low_ilisi(evidence)
        and evidence.get("frac_cells_fov_pure", 0.0) < 0.25
    )


def _rule_decision(evidence: dict) -> tuple[str, str, str]:
    """Deterministic fallback decision."""
    if _batch_signal(evidence):
        return (
            "BATCH_EFFECT",
            "correct_and_recluster",
            f"clustering explained by FOV (CramersV="
            f"{evidence['cramers_v']:.2f}), low iLISI "
            f"({evidence['median_ilisi']:.2f}), "
            f"{evidence['frac_cells_fov_pure']:.0%} of cells in FOV-pure "
            f"clusters",
        )
    if _no_signal(evidence):
        return ("BIOLOGY", "keep", "clusters independent of FOV; neighbors mix")
    return (
        "UNCERTAIN",
        "flag_for_user",
        "partial FOV-association signal (possibly a region-restricted cell "
        "type) — needs human confirmation",
    )


def _guardrail(
    decision: str, action: str, reason: str, evidence: dict
) -> tuple[str, str, str]:
    """Override the LLM if it contradicts the hard metrics."""
    # Under-correction guard: LLM says BIOLOGY but evidence is strong.
    if decision == "BIOLOGY" and _batch_signal(evidence):
        return (
            "UNCERTAIN",
            "flag_for_user",
            "LLM said biology but strong FOV-bias metrics present — "
            "escalating instead of silently keeping",
        )
    # Over-correction guard: LLM says BATCH_EFFECT with no supporting metrics.
    if decision == "BATCH_EFFECT" and _no_signal(evidence):
        return (
            "BIOLOGY",
            "keep",
            "LLM flagged batch effect but metrics are clean — avoiding "
            "over-correction that would erase spatial biology",
        )
    return decision, action, reason


def llm_fov_bias_adjudicate(evidence: dict) -> dict:
    """Decide whether suspicious clusters are FOV batch effect or biology.

    Returns {"decision", "action", "reason", "source"} where source is
    "llm" or "rule".
    """
    llm = _make_llm()
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            print(
                "  [fov-bias] asking LLM to adjudicate cluster pairs...",
                flush=True,
            )
            resp = llm.invoke(
                [
                    SystemMessage(content=_FOV_SYSTEM_PROMPT),
                    HumanMessage(content=_format_evidence(evidence)),
                ]
            )
            text = resp.content.strip()
            dm = re.search(
                r"DECISION:\s*(BATCH_EFFECT|BIOLOGY|UNCERTAIN)",
                text,
                re.IGNORECASE,
            )
            am = re.search(
                r"ACTION:\s*(correct_and_recluster|keep|flag_for_user)",
                text,
                re.IGNORECASE,
            )
            rm = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
            if dm and am:
                decision = dm.group(1).upper()
                action = am.group(1).lower()
                reason = rm.group(1).strip() if rm else text
                decision, action, reason = _guardrail(
                    decision, action, reason, evidence
                )
                return {
                    "decision": decision,
                    "action": action,
                    "reason": reason,
                    "source": "llm",
                }
        except Exception:
            pass  # fall through to rules

    decision, action, reason = _rule_decision(evidence)
    return {
        "decision": decision,
        "action": action,
        "reason": reason,
        "source": "rule",
    }
