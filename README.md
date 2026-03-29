# StratGraph: A Cost-Efficient Graph-RAG Framework via Structural Stratified Sampling

## Overview

This repository contains the official implementation of StratGraph, a novel cost-efficient Graph-RAG framework introduced in the paper "StratGraph: A Cost-Efficient Graph-RAG Framework via Structural Stratified Sampling".StratGraph addresses the systematic "Hub Bias" found in existing centrality-based pruning methods (such as PageRank) by prioritizing structural diversity over global centrality. Through its core Structural Stratified Sampling ($S^3$) strategy, it enforces proportional budget allocation across semantic communities, ensuring the retention of essential "bridging evidence" necessary for reliable multi-hop reasoning.

## Key Features
- Mitigates Hub Bias: Shifts the indexing paradigm from global centrality ranking to structural diversity-aware selection, preventing the systematic suppression of critical evidence from peripheral long-tail semantic regions.
- Structural Stratified Sampling ($S^3$): Integrates community detection (supporting both Geometric and Topological paradigms) with proportional budget allocation and local centrality ranking.
- Dual-Granularity Index: Reconciles the need for high-level reasoning with broad lexical coverage by building a balanced index. This includes a Semantic Layer (Knowledge Graph Skeleton) and a Lexical Layer (Keyword-Bipartite graph).
- Exceptional Cost-Efficiency: Consistently outperforms centrality-based baselines under equivalent budgets. Notably, it surpasses the strongest baseline even when the indexing budget is reduced by 20%.

## Architecture
The StratGraph framework operates in three main phases:
- Topology Initialization: Models latent semantic relationships between text chunks using a lightweight proxy graph.
- Stratified Skeleton Construction: Identifies a budget-constrained set of "Core Chunks" via the $S^3$ algorithm to form the semantic backbone.
- Dual-Granularity Indexing: Supplements the KG skeleton with a fine-grained keyword-bipartite graph to ensure zero information loss.

## Prerequisites

Ensure you have Python **>=3.10** installed.

## Installation

Install dependencies using Poetry:

```bash
pip install poetry
poetry install
```

## Setup

Using the folder `ragtest-musique` as an example, follow these steps:

### 1. Initialize the Project

```bash
python -m graphrag init --root ragtest-musique/
```

This command sets up the necessary file structure and configurations.

### 2. Tune Prompts

```bash
python -m graphrag prompt-tune --root ragtest-musique/ --config ragtest-musique/settings.yaml --discover-entity-types
```

Adjust prompts for better retrieval.

### 3. Build the Index

Before running this step, modify `settings.yaml` to set the appropriate parameters as needed, based on our paper.

```bash
python -m graphrag index --root ragtest-musique/
```

This process creates an indexed structure for retrieval.

## Generate Context and Answers

### Environment Setup

Before executing the scripts, set up your API key:

```bash
export GRAPHRAG_API_KEY=your_api_key_here
```

### 1. Generate Context

To generate a single context file:

```bash
python indexing_sket/create_context.py ragtest-musique/ keyword 0.5
```

#### Parameters:

- **First argument**: Root directory of the project
- **Second argument**: Context-building strategy (`text`, `keyword`, or `skeleton`)
- **Third argument**: Context threshold **theta** (range: `0.0-1.0`)

### 2. Generate Answers

To generate answers for all context files in the output directory:

```bash
python indexing_sket/llm_answer.py ragtest-musique/
```

## Acknowledgments

This project builds upon [Microsoft's GraphRAG (version 0.4.1)](https://github.com/microsoft/graphrag/commit/ba50caab4d2fea9bc3fd926dd9051b9f4cebf6bd), licensed under the MIT License.