# 🧬 BioAI Agent Suite

> **Concept & Domain Expertise:** Dr. Abu Galib (drgalib20)  
> **Technical Implementation:** Claude AI (Anthropic)  
> **Platform:** Ubuntu 24.04 · Python 3.12.3 · HuggingFace Inference API

---

## Overview

A terminal-based biomedical AI research tool that queries multiple Large Language Models (LLMs) simultaneously on clinical prompts, grounded in real-time PubMed literature retrieval.

Designed for **zero-shot clinical decision making research** — comparing how different LLMs respond to medical queries when augmented with current published evidence.

**No local GPU required.** All inference runs via HuggingFace's serverless API.

---

## How It Works

This agent implements **RAG (Retrieval Augmented Generation)** to compensate for general LLMs not being specifically trained on biomedical corpora:

```
Clinical Prompt
      ↓
PubMed Search (NCBI E-utilities API)
      ↓
Abstract Injection into prompt context
      ↓
Parallel LLM querying (all models simultaneously)
      ↓
Side-by-side responses + CSV + MLflow logging
```

> **Key insight:** Large general models (70B parameters) with real-time PubMed grounding often match or outperform smaller biomedical fine-tuned models on clinical QA tasks — combining superior base reasoning with current literature access.

---

## Agents

### `bioai_agent.py` — v1 (General Comparison)
- 3 general LLMs via HF Inference API
- PubMed grounding
- Multi-turn conversation history
- CSV + MLflow logging

### `bioai_agent_biomedical.py` — v2 (Main Agent ★)
- 5 LLMs ranked by biomedical benchmark performance
- Biomedical expert system prompt injected on every query
- Year-tagged PubMed abstracts
- Runtime model enable/disable
- Latency scoreboard
- Word count per response
- CSV + MLflow logging

---

## Models

| # | Model | Size | Benchmark Strength |
|---|---|---|---|
| 1 | `meta-llama/Llama-3.3-70B-Instruct` | 70B | Best reasoning |
| 2 | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Strong medical QA |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | 7B | Biomedical benchmarks |
| 4 | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Lightweight, fast |
| 5 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Baseline |

All models run via **HuggingFace Serverless Inference API** — free tier.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/drgalib20/bioai-agent.git
cd bioai-agent
```

### 2. Create and activate virtual environment
```bash
python3 -m venv bioai-env
source bioai-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your HuggingFace token
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
```bash
export HF_TOKEN=hf_your_token_here
```
To make it permanent:
```bash
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

### 5. Run the agent
```bash
# v2 — Recommended main agent
python bioai_agent_biomedical.py

# v1 — Basic 3-model agent
python bioai_agent.py
```

---

## Usage

Once running, type any clinical or biomedical prompt at the `You >` prompt:

```
You > What are the mechanisms of action of ACE inhibitors in hypertension?
You > Compare metformin and SGLT2 inhibitors for type 2 diabetes management
You > Explain the pathophysiology of septic shock
You > What are first-line treatments for rheumatoid arthritis?
```

### Terminal Commands (v2)

| Command | Description |
|---|---|
| `/models` | List all models with status |
| `/enable <n>` | Enable model by number e.g. `/enable 1` |
| `/disable <n>` | Disable model by number e.g. `/disable 3` |
| `/pubmed` | Toggle PubMed grounding on/off |
| `/history` | Show conversation history |
| `/clear` | Clear conversation history |
| `/score` | Show latency scoreboard |
| `/save` | Show results CSV path |
| `/quit` | Exit |

---

## Output & Logging

Every session automatically logs to:

| File | Contents |
|---|---|
| `~/bioai_results.csv` | v1 results |
| `~/bioai_biomedical_results.csv` | v2 results |

**Columns:** `timestamp, session_id, turn, prompt, model, response, latency_s, word_count, pubmed_pmids`

### MLflow Tracking
```bash
# View experiment dashboard
mlflow ui

# Open in browser
http://localhost:5000
```
Experiments: `bioai_agent` (v1) · `bioai_biomedical_agent` (v2)

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Linux / macOS / Windows (WSL) |
| Python | 3.10+ |
| RAM | 4GB minimum |
| GPU | ❌ Not required |
| Internet | ✅ Required (API calls) |
| HF Account | ✅ Free tier sufficient |

> Developed on: AMD Ryzen 5 5500 · 16GB RAM · AMD RX 560 · Ubuntu 24.04

---

## Architecture

```
bioai-agent/
├── bioai_agent.py                 # v1 — 3-model general agent
├── bioai_agent_biomedical.py      # v2 — 5-model biomedical agent (main)
├── run_bioai.sh                   # Quick launcher script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Limitations

- True biomedical fine-tuned models (BioMistral, Meditron) require local GPU or paid cloud inference — not available on HF free serverless tier
- General models with biomedical system prompts + PubMed RAG are used as a practical alternative
- HuggingFace free tier has rate limits — heavy usage may hit quotas
- PubMed grounding adds ~2-3 seconds per query

---

## Roadmap

- [ ] Replicate API integration (BioMistral-7B via API)
- [ ] Kaggle notebook version (30hr/week free GPU)
- [ ] Automated response quality scoring
- [ ] HTML comparison report generator
- [ ] Meditron-7B (pending gated access approval)

---

## References

- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- [NCBI E-utilities (PubMed API)](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)
- [Meditron-7B](https://huggingface.co/epfl-llm/meditron-7b)

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Concept & Domain Expertise: Dr. Abu Galib · Technical Implementation: Claude AI (Anthropic)*
