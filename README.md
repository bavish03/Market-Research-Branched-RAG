# Branched RAG Market Research Assistant

A Retrieval-Augmented Generation (RAG) application designed for market intelligence and research. Unlike traditional RAG systems that search a single knowledge base, this project uses a Branched RAG architecture to route user queries across multiple specialized knowledge domains.

## Features

* Multi-branch retrieval architecture

  * Company Research Branch
  * Industry Research Branch
  * News Research Branch

* Local AI stack

  * Embedding Model: all-MiniLM-L6-v2
  * LLM: Phi-3 Mini (via Ollama)

* Vector Search

  * ChromaDB for semantic retrieval

* Interactive UI

  * Streamlit-based chat interface
  * Persistent conversation memory
  * Source document viewer
  * Active branch visualization

## Tech Stack

* Python
* Streamlit
* ChromaDB
* LangChain
* Sentence Transformers
* Ollama
* Phi-3 Mini

## Architecture

User Query
→ Branch Router
→ Multi-Branch Retrieval
→ ChromaDB Vector Search
→ Context Aggregation
→ Phi-3 Mini
→ Market Research Analysis

## Data Sources

The system currently retrieves information from:

* Company Annual Reports (PDFs)
* Industry Research Reports (PDFs)
* News Dataset

Documents are chunked, embedded using all-MiniLM-L6-v2, and stored in ChromaDB for semantic retrieval.

This project demonstrates how a complete local RAG pipeline can be built and deployed even on resource-constrained hardware.
