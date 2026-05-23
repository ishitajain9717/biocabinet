"""Shared utilities for the enrichment (GNN-PPI) subpackage.

Exports:
    Metrictor_PPI       — multi-label classification metrics (P/R/F1) for PPI edges
    UnionSet            — union-find for counting connected components in PPI graph
    get_bfs_sub_graph   — pull a BFS-grown subgraph of edges (for train/val split)
    get_dfs_sub_graph   — same, DFS variant
    print_file          — print to stdout AND optionally append to a log file
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch


# ---------- logging helper ----------

def print_file(msg: str, save_file_path: str | None = None) -> None:
    """Print to stdout and optionally append the same message to a log file."""
    print(msg)
    if save_file_path:
        with open(save_file_path, "a") as f:
            f.write(msg + "\n")


# ---------- metrics ----------

class Metrictor_PPI:
    """Multi-label classification metrics for the 7-class PPI task.

    Computes micro-averaged Precision/Recall/F1 across all (edge, class) pairs.
    Vectorized over PyTorch tensors — no Python loops.
    """

    def __init__(self, pred_y: torch.Tensor, true_y: torch.Tensor, is_binary: bool = False):
        pred = pred_y.detach().cpu().to(torch.bool)
        true = true_y.detach().cpu().to(torch.bool)

        if is_binary:
            assert pred.ndim == 1 and true.ndim == 1, "binary: expect 1-D tensors"
        else:
            assert pred.shape == true.shape, f"shape mismatch: {pred.shape} vs {true.shape}"

        self.TP = int(( pred &  true).sum().item())
        self.FP = int(( pred & ~true).sum().item())
        self.FN = int((~pred &  true).sum().item())
        self.TN = int((~pred & ~true).sum().item())
        self.num = pred.numel()

    @property
    def Precision(self) -> float:
        denom = self.TP + self.FP
        return self.TP / denom if denom else 0.0

    @property
    def Recall(self) -> float:
        denom = self.TP + self.FN
        return self.TP / denom if denom else 0.0

    @property
    def F1(self) -> float:
        p, r = self.Precision, self.Recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def show_result(self) -> None:
        print(f"  TP={self.TP}  FP={self.FP}  TN={self.TN}  FN={self.FN}")
        print(f"  Precision={self.Precision:.4f}  Recall={self.Recall:.4f}  F1={self.F1:.4f}")


# ---------- union-find ----------

class UnionSet:
    """Union-find with path compression and union-by-rank.

    Used to count connected components in the PPI graph (sanity check that
    the graph is mostly one connected blob before training).
    """

    def __init__(self, m: int):
        self.roots = list(range(m))
        self.rank = [0] * m
        self.count = m

    def find(self, member: int) -> int:
        path = []
        while member != self.roots[member]:
            path.append(member)
            member = self.roots[member]
        for node in path:
            self.roots[node] = member
        return member

    def union(self, p: int, q: int) -> None:
        rp, rq = self.find(p), self.find(q)
        if rp == rq:
            return
        if self.rank[rp] < self.rank[rq]:
            self.roots[rp] = rq
        elif self.rank[rp] > self.rank[rq]:
            self.roots[rq] = rp
        else:
            self.roots[rq] = rp
            self.rank[rp] += 1
        self.count -= 1


# ---------- BFS / DFS subgraph splitters ----------
# Module-level (despite where the originals lived) — gnn_data.py calls them as such.

def get_bfs_sub_graph(
    ppi_list: Sequence[Sequence[int]],
    node_num: int,
    node_to_edge_index: dict[int, list[int]],
    sub_graph_size: int,
) -> list[int]:
    """Grow a connected subgraph by BFS until it contains `sub_graph_size` edges."""
    seed = _pick_seed_node(node_to_edge_index, node_num)
    queue = deque([seed])
    selected_nodes: set[int] = set()
    selected_edges: list[int] = []
    seen_edges: set[int] = set()

    while queue and len(selected_edges) < sub_graph_size:
        curr = queue.popleft()
        if curr in selected_nodes:
            continue
        selected_nodes.add(curr)
        for eidx in node_to_edge_index.get(curr, []):
            if eidx not in seen_edges:
                seen_edges.add(eidx)
                selected_edges.append(eidx)
                if len(selected_edges) >= sub_graph_size:
                    break
            other = ppi_list[eidx][1] if ppi_list[eidx][0] == curr else ppi_list[eidx][0]
            if other not in selected_nodes:
                queue.append(other)
    return selected_edges


def get_dfs_sub_graph(
    ppi_list: Sequence[Sequence[int]],
    node_num: int,
    node_to_edge_index: dict[int, list[int]],
    sub_graph_size: int,
) -> list[int]:
    """Same as BFS but with a stack — useful for tighter cluster-shaped val sets."""
    seed = _pick_seed_node(node_to_edge_index, node_num)
    stack = [seed]
    selected_nodes: set[int] = set()
    selected_edges: list[int] = []
    seen_edges: set[int] = set()

    while stack and len(selected_edges) < sub_graph_size:
        curr = stack.pop()
        if curr in selected_nodes:
            continue
        selected_nodes.add(curr)
        for eidx in node_to_edge_index.get(curr, []):
            if eidx not in seen_edges:
                seen_edges.add(eidx)
                selected_edges.append(eidx)
                if len(selected_edges) >= sub_graph_size:
                    break
            other = ppi_list[eidx][1] if ppi_list[eidx][0] == curr else ppi_list[eidx][0]
            if other not in selected_nodes:
                stack.append(other)
    return selected_edges


def _pick_seed_node(
    node_to_edge_index: dict[int, list[int]],
    node_num: int,
    max_tries: int = 100,
) -> int:
    """Pick a low-degree starting node so the subgraph isn't dominated by one hub."""
    for _ in range(max_tries):
        candidate = int(np.random.randint(0, node_num))
        if 0 < len(node_to_edge_index.get(candidate, [])) <= 5:
            return candidate
    return next(iter(node_to_edge_index))
