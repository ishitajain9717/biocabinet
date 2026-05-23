"""End-to-end multi-sample preprocessing runner."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.bulk_rnaseq.config import PreprocessingConfig, Sample, collect_config_from_user
from scripts.bulk_rnaseq.nodes import (
    node_align,
    node_fastqc,
    node_featurecounts,
    node_normalize,
    node_trim,
)
from scripts.common.node_result import NodeResult


def _cfg_to_jsonable(cfg: PreprocessingConfig) -> dict:
    raw = asdict(cfg)

    def _conv(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_conv(i) for i in v]
        if isinstance(v, dict):
            return {k2: _conv(v2) for k2, v2 in v.items()}
        return v

    return {k: _conv(v) for k, v in raw.items()}


def _run_sample(
    cfg: PreprocessingConfig, sample: Sample
) -> tuple[list[NodeResult], Path | None]:
    """Run all per-sample nodes. Returns (history, counts_txt | None)."""
    history: list[NodeResult] = []

    qc = node_fastqc(cfg, sample)
    history.append(qc)
    if not qc.ok:
        return history, None

    tr = node_trim(cfg, sample)
    history.append(tr)
    if not tr.ok:
        return history, None

    r1 = Path(tr.outputs["r1"])
    r2 = Path(tr.outputs["r2"]) if "r2" in tr.outputs else None

    al = node_align(cfg, sample, r1, r2)
    history.append(al)
    if not al.ok:
        return history, None

    bam_path = Path(al.outputs["bam"])
    fc = node_featurecounts(cfg, sample, bam_path)
    history.append(fc)
    if not fc.ok:
        return history, None

    return history, Path(fc.outputs["counts_txt"])


def main() -> int:
    cfg = collect_config_from_user()

    per_sample_histories: dict[str, list[NodeResult]] = {}
    count_results: list[tuple[str, Path]] = []
    failed: list[str] = []

    for sample in cfg.samples:
        print(f"\n--- Sample: {sample.name} ---")
        history, counts_txt = _run_sample(cfg, sample)
        per_sample_histories[sample.name] = history
        for r in history:
            status = "OK  " if r.ok else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")
        if counts_txt is None:
            failed.append(sample.name)
            print(f"  => stopped early for this sample")
        else:
            count_results.append((sample.name, counts_txt))

    norm_result: NodeResult | None = None
    if count_results:
        print(f"\n--- Normalization {cfg.normalizations} ---")
        norm_result = node_normalize(cfg, count_results)
        status = "OK  " if norm_result.ok else "FAIL"
        print(f"  [{status}] {norm_result.message}")
    else:
        print("\nNo successful samples — skipping normalization.")

    payload = {
        "config": _cfg_to_jsonable(cfg),
        "samples": {
            name: [r.to_dict() for r in hist]
            for name, hist in per_sample_histories.items()
        },
        "normalize": norm_result.to_dict() if norm_result else None,
        "failed_samples": failed,
    }
    report = cfg.out_dir / "preprocessing_run.json"
    report.write_text(json.dumps(payload, indent=2))
    print(f"\nReport written to: {report}")

    all_ok = (not failed) and (norm_result is not None) and norm_result.ok
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
