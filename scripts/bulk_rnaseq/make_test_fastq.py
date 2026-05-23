"""Generate a small synthetic paired-end FASTQ from a reference FASTA.

Smoke-test data only; not a biological simulator.
"""
from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    name: str | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def write_pair(r1, r2, n: int, ref: str, read_len: int,
               frag_min: int, frag_max: int, seed: int) -> int:
    rng = random.Random(seed)
    qual = "I" * read_len
    valid = set("ACGT")
    written = 0
    attempts = 0
    while written < n and attempts < n * 50:
        attempts += 1
        frag = rng.randint(frag_min, frag_max)
        if frag > len(ref):
            continue
        start = rng.randint(0, len(ref) - frag)
        seq = ref[start:start + frag]
        if any(c not in valid for c in seq):
            continue
        f1 = seq[:read_len]
        f2 = revcomp(seq[-read_len:])
        rid = f"read{written + 1}"
        r1.write(f"@{rid}/1\n{f1}\n+\n{qual}\n")
        r2.write(f"@{rid}/2\n{f2}\n+\n{qual}\n")
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample", default="testsample")
    ap.add_argument("--reads", type=int, default=5000)
    ap.add_argument("--read-len", type=int, default=100)
    ap.add_argument("--frag-min", type=int, default=200)
    ap.add_argument("--frag-max", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.fasta} ...")
    seqs = read_fasta(args.fasta)
    if not seqs:
        raise SystemExit(f"No sequences in {args.fasta}")
    name, ref = next(iter(seqs.items()))
    print(f"Using contig {name} (len={len(ref):,})")

    r1_path = args.out_dir / f"{args.sample}_R1.fastq.gz"
    r2_path = args.out_dir / f"{args.sample}_R2.fastq.gz"
    with gzip.open(r1_path, "wt") as r1, gzip.open(r2_path, "wt") as r2:
        n = write_pair(r1, r2, args.reads, ref, args.read_len,
                       args.frag_min, args.frag_max, args.seed)

    if n < args.reads:
        print(f"WARNING: produced {n}/{args.reads} reads (chromosome had many N stretches)")
    print(f"Wrote {r1_path}")
    print(f"Wrote {r2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
