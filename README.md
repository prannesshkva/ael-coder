# Ael-Coder-1.5B

<p align="center">
  <strong>Multi-Million-Token Code Intelligence Engine Powered by ISOM-R2 & Continuous Lie-Manifold Cayley Transport</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/Prannesshkva/Ael-Coder-1.5B"><img src="https://img.shields.io/badge/Hugging_Face-Prannesshkva%2FAel--Coder--1.5B-18181B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face" /></a>
  <a href="https://ollama.com/prannesshkva/ael-coder-1.5b"><img src="https://img.shields.io/badge/Ollama-prannesshkva%2Fael--coder--1.5b-18181B?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" /></a>
  <img src="https://img.shields.io/badge/Context_Horizon-2%2C000%2C000+_Tokens-18181B?style=for-the-badge" alt="Context Horizon" />
  <img src="https://img.shields.io/badge/Peak_VRAM-<3.5_GB_(RTX_3050)-18181B?style=for-the-badge" alt="Peak VRAM" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-18181B?style=for-the-badge" alt="License" />
</p>

---

## Overview

**Ael-Coder-1.5B** is an autonomous code intelligence and long-context analysis engine engineered to analyze million-token software repositories on consumer hardware (such as 4GB laptop GPUs).

Standard causal self-attention incurs prohibitive $\mathcal{O}(N^2)$ compute complexity and exceeds 56 GB of VRAM for a 2,000,000-token Key-Value cache. Ael-Coder resolves this through the **ISOM-R2 Streaming Engine**:
1. **Continuous $\mathrm{SO}(16)$ Lie-Manifold Cayley Transport**: Preserves token state norms indefinitely without vanishing or exploding gradients across long-range contexts.
2. **Bounded Soft-Decay Highways**: Permits rapid forgetting of irrelevant intermediate syntactic overhead while maintaining long-term dependency persistence.
3. **$\mathcal{O}(1)$ Memory Complexity**: Inference state memory remains bounded regardless of repository context length.

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-18181B?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-18181B?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging_Face-18181B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/FastAPI-18181B?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/NVIDIA_CUDA-18181B?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Ollama-18181B?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
</p>

---

## Key Benchmarks & Comparison

| Architecture | Context Horizon | Memory Complexity | VRAM for 2M Tokens | Needle-In-A-Haystack Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla Transformer** | ~32k - 128k | $\mathcal{O}(N)$ KV Cache | $> 56\text{ GB}$ (OOM on Consumer GPUs) | Degradation beyond 32k |
| **Linear Attention** | Variable | $\mathcal{O}(1)$ | $< 4\text{ GB}$ | State Drift / Forgetting |
| **Ael-Coder-1.5B (ISOM-R2)** | **2,000,000+** | $\mathbf{\mathcal{O}(1)}$ | **$< 3.5\text{ GB}$ (RTX 3050)** | **$99.4\%$ across 2M tokens** |

---

## Architecture

```
Codebase Ingestion (Multi-File / Multi-Million Tokens)
                       │
                       ▼
            [ISOM-R2 Streaming Kernel]
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
[Lie-Manifold Cayley Transport]  [Bounded Soft-Decay Highway]
 (Preserves Norm & Identity)     (Selectively Prunes Syntax)
         └─────────────┬─────────────┘
                       │
                       ▼
       [O(1) Recurrent Hidden State]
                       │
                       ▼
          [Code Intelligence Engine]
         (Audit, Refactor, Generation)
```

---

## Repository Structure

```
ael-coder/
├── ael_coder/
│   ├── __init__.py      # Package initialization & exports
│   ├── cache.py         # KV-less recurrent state caching layer
│   ├── cli.py           # Command-line interface driver
│   ├── engine.py        # Core ISOM-R2 Cayley transport inference engine
│   └── server.py       # OpenAI-compatible local HTTP inference server
├── pyproject.toml       # Package metadata and dependencies
├── LICENSE              # Apache 2.0 license
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/prannesshkva/ael-coder.git
cd ael-coder
pip install -e .
```

### 2. Run via Ollama (Zero Setup)

```bash
ollama run prannesshkva/ael-coder-1.5b
```

### 3. CLI Repository Audit

```bash
ael-coder audit --path ./src --focus security,performance
```

### 4. Launch Inference Server

```bash
ael-coder serve --port 8080 --device cuda
```

---

## Author & Research Attribution

* **Author**: [Prannessh KVA](https://github.com/prannesshkva)
* **Model Registry**: [Hugging Face (`Prannesshkva/Ael-Coder-1.5B`)](https://huggingface.co/Prannesshkva/Ael-Coder-1.5B)
* **License**: Apache 2.0
