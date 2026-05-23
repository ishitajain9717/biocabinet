"""GNNDataset: load STRING PPI + ESM-2 embeddings + (optional) pathway embeddings.

Builds a PyG Data object ready for GIN training:
    x:           (N_proteins, esm_dim)        — frozen ESM-2 mean-pooled per protein
    pathway_x:   (N_proteins, 768)            — BioBERT pathway embeddings (zero if missing)
    edge_index:  (2, 2E)                      — undirected (forward + reverse stacked)
    edge_attr:   (2E, 7)                      — multi-label class flags

Multi-label aggregation: a single (p1, p2) pair may appear with multiple modes
in the STRING file (e.g. binding AND reaction). We OR them into a 7-dim binary vector.

Edge layout convention: edge_index[:, :E] are the "forward" edges; edge_index[:, E:]
are the same edges in reverse. train_mask / val_mask index into [0, E) — the model
samples one direction per unique pair (the labels are symmetric anyway).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data

from scripts.enrichment.utils import (
    UnionSet,
    get_bfs_sub_graph,
    get_dfs_sub_graph,
)


CLASS_MAP = {
    "reaction":   0,
    "binding":    1,
    "ptmod":      2,
    "activation": 3,
    "inhibition": 4,
    "catalysis":  5,
    "expression": 6,
}
NUM_CLASSES = len(CLASS_MAP)


class GNNDataset:
    """Self-contained PPI dataset wrapper.

    On instantiation it walks every input file once, builds the tensors, and
    is then ready to be split (`split(...)`) and converted to PyG (`to_pyg_data()`).
    """

    def __init__(
        self,
        ppi_path:         str | Path,
        esm_emb_path:     str | Path,
        pathway_emb_path: Optional[str | Path] = None,
        exclude_path:     Optional[str | Path] = None,
        verbose:          bool = True,
    ):
        self.ppi_path         = Path(ppi_path)
        self.esm_emb_path     = Path(esm_emb_path)
        self.pathway_emb_path = Path(pathway_emb_path) if pathway_emb_path else None
        self.exclude_path     = Path(exclude_path) if exclude_path else None
        self.verbose          = verbose

        self.protein_to_idx:     dict[str, int] = {}
        self.idx_to_protein:     dict[int, str] = {}
        self.edge_list:          list[tuple[int, int]] = []
        self.edge_label_list:    list[list[int]] = []
        self.protein_to_esm:     dict[str, torch.Tensor] = {}
        self.protein_to_pathway: dict[str, torch.Tensor] = {}
        self.esm_dim:            int = 0

        self.x:          Optional[torch.Tensor] = None
        self.pathway_x:  Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr:  Optional[torch.Tensor] = None

        self.train_mask: list[int] = []
        self.val_mask:   list[int] = []

        self._build()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _build(self) -> None:
        self._parse_ppi()
        self._load_esm_embeddings()
        if self.pathway_emb_path is not None:
            self._load_pathway_embeddings()
        self._build_node_features()
        self._build_edges()
        self._log(
            f"GNNDataset ready: N={self.num_nodes} proteins, "
            f"E={self.num_unique_edges} unique undirected edges "
            f"({self.num_edges} directed in edge_index)"
        )

    # ---------- properties ----------

    @property
    def num_nodes(self) -> int:
        return len(self.protein_to_idx)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index is not None else 0

    @property
    def num_unique_edges(self) -> int:
        return self.num_edges // 2

    # ---------- parsers ----------

    def _parse_ppi(self) -> None:
        excluded: set[str] = set()
        if self.exclude_path is not None:
            with self.exclude_path.open() as f:
                excluded = set(json.load(f))

        edge_dict:    dict[str, int]   = {}
        edge_labels:  list[list[int]]  = []
        protein_to_idx: dict[str, int] = {}

        with self.ppi_path.open() as f:
            f.readline()  # skip header line
            for line in tqdm(f, desc="parsing PPI", disable=not self.verbose):
                parts = line.rstrip().split("\t")
                if len(parts) < 3:
                    continue
                p1, p2, mode = parts[0], parts[1], parts[2]
                if p1 in excluded or p2 in excluded:
                    continue
                if mode not in CLASS_MAP:
                    continue
                key = f"{p1}__{p2}" if p1 < p2 else f"{p2}__{p1}"
                if key not in edge_dict:
                    edge_dict[key] = len(edge_labels)
                    edge_labels.append([0] * NUM_CLASSES)
                    if p1 not in protein_to_idx:
                        protein_to_idx[p1] = len(protein_to_idx)
                    if p2 not in protein_to_idx:
                        protein_to_idx[p2] = len(protein_to_idx)
                edge_labels[edge_dict[key]][CLASS_MAP[mode]] = 1

        self.protein_to_idx  = protein_to_idx
        self.idx_to_protein  = {i: p for p, i in protein_to_idx.items()}
        self.edge_label_list = edge_labels

        for key in edge_dict:
            p1, p2 = key.split("__")
            self.edge_list.append((protein_to_idx[p1], protein_to_idx[p2]))

        self._log(
            f"  parsed {len(self.edge_list)} unique undirected pairs, "
            f"{len(protein_to_idx)} proteins"
        )

    def _load_esm_embeddings(self) -> None:
        cache = torch.load(self.esm_emb_path, weights_only=False)
        if not isinstance(cache, dict) or not cache:
            raise ValueError(f"ESM embedding file is empty / not a dict: {self.esm_emb_path}")
        self.protein_to_esm = cache
        sample_vec = next(iter(cache.values()))
        self.esm_dim = sample_vec.shape[0]
        self._log(
            f"  loaded ESM embeddings: {len(cache)} proteins, dim={self.esm_dim}, "
            f"from {self.esm_emb_path.name}"
        )

    def _load_pathway_embeddings(self) -> None:
        embeds = torch.load(self.pathway_emb_path, weights_only=False)
        for k, v in embeds.items():
            self.protein_to_pathway[k] = v
            self.protein_to_pathway[f"9606.{k}"] = v
        self._log(f"  loaded pathway embeddings for {len(embeds)} unique ENSPs")

    # ---------- feature builders ----------

    def _build_node_features(self) -> None:
        N = self.num_nodes
        D = self.esm_dim

        x = torch.zeros(N, D, dtype=torch.float32)
        n_with_esm = 0
        n_missing  = 0
        for protein, idx in self.protein_to_idx.items():
            emb = self.protein_to_esm.get(protein)
            if emb is None:
                bare = protein.split(".", 1)[-1] if "." in protein else protein
                emb = self.protein_to_esm.get(bare)
            if emb is None:
                n_missing += 1
                continue
            x[idx] = emb.float()
            n_with_esm += 1
        self.x = x
        self._log(
            f"  built x: shape={tuple(x.shape)}, "
            f"{n_with_esm}/{N} proteins had ESM embeddings"
            + (f", {n_missing} missing (zero-filled)" if n_missing else "")
        )

        if self.pathway_emb_path is not None and self.protein_to_pathway:
            pw_dim = next(iter(self.protein_to_pathway.values())).shape[0]
            pw = torch.zeros(N, pw_dim, dtype=torch.float32)
            n_with_pw = 0
            for protein, idx in self.protein_to_idx.items():
                emb = self.protein_to_pathway.get(protein)
                if emb is None:
                    bare = protein.split(".", 1)[-1] if "." in protein else protein
                    emb = self.protein_to_pathway.get(bare)
                if emb is not None:
                    pw[idx] = emb
                    n_with_pw += 1
            self.pathway_x = pw
            self._log(
                f"  built pathway_x: shape={tuple(pw.shape)}, "
                f"{n_with_pw}/{N} proteins had pathway embeddings"
            )

    def _build_edges(self) -> None:
        forward = self.edge_list
        n_unique = len(forward)

        edge_index = torch.empty(2, 2 * n_unique, dtype=torch.long)
        for k, (i, j) in enumerate(forward):
            edge_index[0, k] = i
            edge_index[1, k] = j
            edge_index[0, n_unique + k] = j
            edge_index[1, n_unique + k] = i
        self.edge_index = edge_index

        edge_attr = torch.tensor(
            self.edge_label_list + self.edge_label_list,
            dtype=torch.float32,
        )
        self.edge_attr = edge_attr

    # ---------- public API ----------

    def to_pyg_data(self) -> Data:
        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )
        if self.pathway_x is not None:
            data.pathway_x = self.pathway_x
        return data

    def count_components(self) -> int:
        ufs = UnionSet(self.num_nodes)
        for i, j in self.edge_list:
            ufs.union(i, j)
        return ufs.count

    def split(
        self,
        method:    str = "random",
        test_size: float = 0.2,
        save_path: Optional[str | Path] = None,
        seed:      int = 42,
    ) -> dict:
        """Split unique-edge index space into train/val.

        method: 'random' | 'bfs' | 'dfs'.
            random — uniform shuffle (easiest, most leaky)
            bfs/dfs — pull a connected subgraph for val (more honest)

        Returned indices are into [0, E) where E = num_unique_edges.
        Equivalent forward (and reverse) edges live at edge_index[:, k]
        and edge_index[:, k + E].
        """
        rng = np.random.default_rng(seed)
        n_unique = len(self.edge_list)
        n_val = int(n_unique * test_size)

        if method == "random":
            perm = rng.permutation(n_unique).tolist()
            self.val_mask   = perm[:n_val]
            self.train_mask = perm[n_val:]
        elif method in {"bfs", "dfs"}:
            node_to_edges: dict[int, list[int]] = {}
            for k, (i, j) in enumerate(self.edge_list):
                node_to_edges.setdefault(i, []).append(k)
                node_to_edges.setdefault(j, []).append(k)
            picker = get_bfs_sub_graph if method == "bfs" else get_dfs_sub_graph
            self.val_mask   = picker(self.edge_list, self.num_nodes, node_to_edges, n_val)
            val_set = set(self.val_mask)
            self.train_mask = [k for k in range(n_unique) if k not in val_set]
        else:
            raise ValueError(f"unknown split method: {method}")

        split_dict = {
            "method":      method,
            "test_size":   test_size,
            "n_train":     len(self.train_mask),
            "n_val":       len(self.val_mask),
            "train_index": self.train_mask,
            "valid_index": self.val_mask,
        }
        if save_path is not None:
            Path(save_path).write_text(json.dumps(split_dict))
        self._log(
            f"  split ({method}): train={len(self.train_mask)}, val={len(self.val_mask)}"
        )
        return split_dict
