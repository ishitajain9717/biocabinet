# rnaseq-agent

An agentic transcriptomic analysis pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph).  
It orchestrates FastQC → trimming → alignment → quantification → normalisation → differential expression → GNN-based PPI enrichment, with an optional RAG layer that grounds LLM summaries in KEGG and Reactome pathway knowledge. It can also deal with spatial transcriptomics and scRNA data. 

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
git clone https://github.com/<your-handle>/rnaseq-agent.git
cd rnaseq-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm]"          # includes OpenAI + Ollama LLM support
```

### External bioinformatics tools

The bulk RNA-seq pipeline calls external tools that must be on your `PATH`:

| Tool | Purpose | Install |
|---|---|---|
| [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) | QC | `brew install fastqc` |
| [Trimmomatic](http://www.usadellab.org/cms/?page=trimmomatic) | Adapter trimming | download JAR |
| [STAR](https://github.com/alexdobin/STAR) | Alignment | `brew install star` |
| [featureCounts](https://subread.sourceforge.net/) | Quantification | `brew install subread` |

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
