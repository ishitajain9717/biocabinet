"""RAG (Retrieval-Augmented Generation) layer for the rnaseq pipeline.

Builds a searchable library of biological pathway descriptions and gene
context, and uses it at inference time to ground LLM answers in trusted
literature sources with citations.
"""
