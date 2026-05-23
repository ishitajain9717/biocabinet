# rnaseq-agent

An agentic RNA-seq analysis pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph).
It orchestrates FastQC → trimming → alignment → quantification → normalisation → differential expression → GNN-based PPI enrichment, with an optional RAG layer that grounds LLM summaries in KEGG and Reactome pathway knowledge.

---

## Architecture

```
Orchestrator (LangGraph)
│
├── Bulk RNA-seq sub-pipeline
│   ├── FastQC  →  Trimmomatic  →  STAR  →  featureCounts
│   ├── Normalisation  (TPM / FPKM / RPKM)
│   ├── Differential Expression  (PyDESeq2)
│   └── Summarise  ← RAG-augmented (Phase 4a)
│
├── scRNA-seq sub-pipeline
│   ├── Load  (h5ad / 10x / pbmc3k)
│   ├── QC  →  Filter  →  Normalise  →  PCA
│   ├── Cluster  (Leiden / UMAP)
│   └── Marker genes  →  Summarise  ← RAG-augmented
│
├── Enrichment sub-pipeline  (auto-chained after bulk)
│   ├── Load PPI graph  (STRING / SHS27k)
│   ├── GNN training  (GIN, BFS split)
│   ├── Evaluation  (test1/2/3 buckets)
│   └── Inference  →  Summarise  ← RAG-augmented
│
└── RAG Q&A session  (Phase 4b — interactive, after final report)
    └── Retriever  →  LLM synthesis  →  Citations
```

Graph diagrams (Mermaid + PNG) live in `docs/`.

---

## Installation

```bash
git clone https://github.com/ishitajain9717/biocabinet.git
cd biocabinet
python -m venv .venv && source .venv/bin/activate

# Option A: editable install (recommended — registers the `rnaseq-agent` CLI)
pip install -e ".[llm]"

# Option B: plain requirements files
pip install -r requirements.txt -r requirements-llm.txt
```

### Python dependencies (all modalities)

Installed via `pip install -e ".[llm]"` from `pyproject.toml`. These are **imported in code** — no separate binary on `PATH`.

| Package | Used by | Role |
|---|---|---|
| `langgraph`, `langchain-core` | Orchestrator, all graphs | Agent workflow + state |
| `pandas`, `numpy`, `scipy` | Bulk, scRNA, enrichment | Tables and numerics |
| `pydeseq2` | Bulk | Differential expression (DESeq2) |
| `scanpy` | scRNA | Load h5ad/10x, QC, normalize, PCA, cluster, markers |
| `torch`, `torch-geometric` | Enrichment | GNN training and inference |
| `transformers` | Enrichment, RAG | BioBERT pathway text embeddings |
| `fair-esm` | Enrichment | ESM-2 protein sequence embeddings |
| `mygene` | Bulk (DEG → enrichment) | ENSG → ENSP mapping |
| `requests`, `tenacity` | Enrichment (pathway build) | KEGG / Reactome / WikiPathways APIs |
| `langchain-openai`, `langchain-ollama` | Summarise, RAG (optional) | LLM summaries and Q&A |

scRNA clustering (Leiden / UMAP) runs inside Scanpy. If Leiden fails at runtime, install graph backends:

```bash
pip install leidenalg python-igraph
```

---

## Requirements by modality

### Bulk RNA-seq

**Workflow:** FASTQ → FastQC → Trimmomatic (optional) → STAR → featureCounts → TPM/FPKM/RPKM → PyDESeq2 → summarise (+ RAG).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) | QC | `brew install fastqc` |
| | [STAR](https://github.com/alexdobin/STAR) | Alignment | `brew install star` |
| | [featureCounts](https://subread.sourceforge.net/) (Subread) | Gene counts | `brew install subread` |
| | [Trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic) | Adapter trim | JAR + `java`; or set `skip_trim=True` |
| | `java` | Runs Trimmomatic | Required only if trimming is enabled |
| **Python (`pip`)** | `pydeseq2`, `pandas`, `mygene` | DEG + pair export for enrichment | In `pyproject.toml` |
| **Reference data** | STAR genome index + GTF | Not in git | See [`data/README.md`](data/README.md) — GENCODE GRCh38 + `STAR --runMode genomeGenerate` |
| **Input** | Paired FASTQ + sample sheet | | Conditions for DESeq2 (≥2 per group) |
| **Optional** | LLM env vars | RAG-augmented summary | See [LLM configuration](#llm-configuration) |

---

### scRNA-seq

**Workflow:** Load (h5ad / 10x / pbmc3k) → QC metrics → filter → normalize → HVG → PCA → neighbors → Leiden → UMAP → marker genes → summarise (+ RAG).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Entire pipeline is Scanpy in Python |
| **Python (`pip`)** | `scanpy`, `numpy`, `scipy`, `pandas` | Core analysis | In `pyproject.toml` |
| | `leidenalg`, `python-igraph` | Leiden clustering | Often needed; install if clustering errors |
| **Input** | `.h5ad`, 10x folder, or `pbmc3k` | | Set in `ScrnaConfig` / interactive prompts |
| **Optional** | `data/pbmc3k_raw.h5ad` | Cached demo data | Download via Scanpy; see `data/README.md` |
| **Optional** | LLM + RAG index | Marker-gene pathway context in summary | Build RAG index once; see RAG section |

---

### Spatial transcriptomics

**Status:** stub only (`scripts/spatial/graph.py` returns “not implemented”). No tools or data required yet.

Planned stack (for reference): spatial `.h5ad` with coordinates, spatial-aware QC/normalization, neighborhood graphs — likely **Scanpy + spatial extensions** (e.g. Squidpy), still **no separate CLI tools** unless we add image registration later.

---

### Enrichment (GNN-PPI)

**Workflow:** Load PPI graph + ESM-2 + pathway embeddings → train GIN (BFS split) → bucketed eval → inference on DEG pairs. Auto-chained after a successful bulk run from the orchestrator.

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Training and inference are PyTorch |
| **Python (`pip`)** | `torch`, `torch-geometric`, `transformers`, `fair-esm` | Model + embeddings | In `pyproject.toml` |
| **Reference data** | `data/ppi_SHS27k.tsv`, `data/esm2_embeddings_SHS27k.pt`, `data/pathway/06_pathway_embeddings_combined.pt` | SHS27k benchmark | See `data/README.md` |
| **Upstream** | Bulk DEG `deg_pairs.tsv` | Optional | Orchestrator passes path when bulk DEG ran |
| **Optional** | GPU (`device=cuda` / `mps`) | Faster training | Default config uses `auto` |

Pathway embedding build (one-time, before merge):

```bash
python -m scripts.enrichment.pathway_embed
python -m scripts.enrichment.reactome_embed
python -m scripts.enrichment.wikipathways_embed
python -m scripts.enrichment.merge_pathway_embeddings
python -m scripts.enrichment.precompute_esm
```

---

### RAG (pathway Q&A)

**Workflow:** Build index once → retriever (BioBERT cosine search) → answerer (LLM + citations) → augment pipeline summaries (4a) → optional interactive chat after orchestrator (4b).

| Type | Requirement | Notes |
|---|---|---|
| **CLI on `PATH`** | *(none)* | Search and embed in Python |
| **Python (`pip`)** | `transformers`, `torch` | BioBERT at index build and query time | In `pyproject.toml` |
| | `langchain-openai` or `langchain-ollama` | Prose answers | Optional; deterministic fallback without LLM |
| **Reference data** | KEGG + Reactome text under `data/pathway/` | Source for `build_index` | Built by enrichment pathway scripts |
| **Generated** | `data/rag/docs.jsonl`, `embeddings.npy`, `gene_to_doc_ids.json` | Local library | Not committed; run `build_index` |

```bash
python -m scripts.rag.build_index          # once, after pathway data exists
python -m scripts.rag.retriever "cell cycle CDK4"   # smoke-test retrieval
python -m scripts.rag.graph                  # interactive Q&A
```

Bulk runs can set **pathway interests** at config time (e.g. `cell cycle, apoptosis`); matching DEGs are always included in RAG gene selection before the top-60% rule.

---

### Orchestrator

| Type | Requirement | Notes |
|---|---|---|
| **Python (`pip`)** | All of the above for chosen modality | Lazy-imports child graphs |
| **Persistence** | `pipeline_runs.sqlite` | LangGraph `SqliteSaver`; created on first run |
| **LLM** | `OLLAMA_MODEL` or `OPENAI_API_KEY` | Final report + RAG chat; optional |

Choosing `bulk_rnaseq` at the prompt runs bulk → enrichment (if bulk succeeds) → final report → optional RAG Q&A. Choosing `scrna` runs scRNA only, then report and optional RAG.

---

## Data setup

See [`data/README.md`](data/README.md) for instructions on downloading the required reference files, PPI graph, and pre-computed embeddings.

---

## Running

### Full orchestrated pipeline

```bash
python -m scripts.orchestrator
# resume a previous run
python -m scripts.orchestrator --thread-id orchestrator_20260520_094200
```

### Individual sub-pipelines

```bash
# Bulk RNA-seq
python -m scripts.bulk_rnaseq.graph

# scRNA-seq
python -m scripts.scrna.graph

# Enrichment (GNN-PPI)
python -m scripts.enrichment.graph

# Build RAG index (run once)
python -m scripts.rag.build_index

# Interactive RAG Q&A
python -m scripts.rag.graph
```

---

## LLM configuration

The pipeline uses an LLM for plain-English summaries and RAG-grounded answers.
Configure via environment variables — no API keys in code:

```bash
# Option A: local Ollama (free, offline)
export OLLAMA_MODEL=llama3

# Option B: OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional, default is gpt-4o-mini
```

If neither is set, all summarise nodes fall back to a deterministic text summary.

---

## Project layout

```
scripts/
├── orchestrator.py          top-level LangGraph orchestrator
├── bulk_rnaseq/             bulk RNA-seq sub-pipeline
│   ├── config.py            PreprocessingConfig dataclass + CLI prompts
│   ├── nodes.py             core bioinformatics functions
│   ├── graph_nodes.py       LangGraph wrapper nodes
│   ├── graph_state.py       PipelineState TypedDict
│   └── graph.py             graph builder
├── scrna/                   scRNA-seq sub-pipeline  (same structure)
├── spatial/                 spatial transcriptomics stub
├── enrichment/              GNN-PPI enrichment sub-pipeline
│   ├── pathway_embed.py     KEGG pathway embeddings (BioBERT)
│   ├── reactome_embed.py    Reactome pathway embeddings
│   ├── wikipathways_embed.py WikiPathways embeddings
│   ├── merge_pathway_embeddings.py  multi-source merge + dedup
│   ├── precompute_esm.py    ESM-2 protein sequence embeddings
│   ├── gnn_data.py          PPI graph + feature assembly
│   ├── gnn_model.py         GIN model definition
│   ├── gnn_train.py         training loop (BFS split)
│   ├── gnn_test.py          bucketed evaluation (test1/2/3)
│   └── inference.py         predict interactions for novel pairs
├── rag/                     RAG layer
│   ├── build_index.py       Phase 1 — build pathway document library
│   ├── retriever.py         Phase 2 — cosine similarity search
│   ├── answerer.py          Phase 3 — LLM synthesis with citations
│   ├── augment.py           Phase 4a — auto-augment pipeline summaries
│   ├── graph_state.py       Phase 4b — RagChatState
│   ├── graph_nodes.py       Phase 4b — chat loop nodes
│   └── graph.py             Phase 4b — interactive Q&A graph
└── common/
    └── node_result.py       NodeResult dataclass (shared across pipelines)
```

---

## Checkpointing and resumption

All pipeline state is persisted to `pipeline_runs.sqlite` via LangGraph's `SqliteSaver`.
If a run is interrupted, resume it by passing the printed `--thread-id`.
Child pipelines (bulk, enrichment) share the same checkpointer as the orchestrator, so partial child runs are also resumable.

---

## License

MIT
