# 🧬 BioAI Agent Suite

> **Concept & Domain Expertise:** Dr. Abu Galib (drgalib20)
> **Technical Implementation:** Claude AI (Anthropic)
> **Platform:** Ubuntu 24.04 · Python 3.12.3 · HuggingFace Inference API

---

## Overview

A terminal-based biomedical AI research tool implementing **Zero-Shot Retrieval Augmented Generation (RAG)** for clinical decision making research.

The suite queries multiple Large Language Models (LLMs) simultaneously on clinical prompts, grounded in real-time evidence retrieved from PubMed and PubMed Central (PMC) free full text. No local GPU is required — all inference runs via HuggingFace Serverless Inference API.

---

## Scientific Design

### Prompting Methodology — Zero-Shot RAG

All three agents implement **Zero-Shot RAG**:

```
Zero-Shot  →  No task examples provided at inference time
RAG        →  Real-time evidence retrieved and injected per query
```

The biomedical system prompt used in v2 and v3 is classified as **expert framing** — it activates and surfaces biomedical knowledge already present in the model from training. It does not constitute few-shot prompting as no Q&A examples are provided.

```
User prompt
    + PubMed/PMC evidence (RAG)              ← new external knowledge
    + Biomedical system prompt (v2/v3 only)  ← activation of existing knowledge
    ↓
LLM → Response
```

### Why Full Text Matters for Clinical Decision Making

Abstract-only evidence is insufficient for clinical decisions. Critical information found only in full text includes:

- Subgroup effects (e.g. efficacy only in EF < 40%, not preserved EF)
- Safety signals and adverse event details
- Dosing nuances and titration protocols
- Methodological limitations affecting result interpretation
- Secondary endpoints and exploratory analyses

Full text from PMC is freely available for ~40% of PubMed articles, covering most NIH-funded and open-access research.

### Evidence Retrieval Pipeline

```
Clinical Prompt
      ↓
PubMed esearch  →  PMIDs (ranked by relevance)
      ↓
PubMed efetch   →  Abstracts + metadata (title, year, DOI)
      ↓
NCBI ID converter  →  PMCID lookup per article
      ↓
PMC efetch      →  Full text XML (if available)
      ↓
Section parser  →  Results → Discussion → Conclusions (priority order)
      ↓
Fallback        →  Abstract if not in PMC
      ↓
Context injection  →  Prepended to LLM prompt
      ↓
LLM response grounded in current published evidence
```

---

## Agent Suite

### Three agents are provided — each serving a distinct research purpose:

| Agent | File | Models | Full Text | System Prompt | Purpose |
|---|---|---|---|---|---|
| **v1** | `bioai_agent_v1.py` | 3 | ✅ PMC | ❌ | Baseline — raw model capability |
| **v2** | `bioai_agent_biomedical.py` | 5 | ❌ Abstract | ✅ | System prompt contribution |
| **v3** | `bioai_agent_v3.py` | 5 | ✅ PMC | ✅ | Full pipeline — clinical research ★ |

### Scientific value of the three-way comparison

```
v1 vs v3  →  Isolates the contribution of the biomedical system prompt
             (same full-text evidence, different framing)

v2 vs v3  →  Isolates the contribution of full-text vs abstract-only evidence
             (same system prompt, different evidence depth)

v1 vs v2  →  Isolates both variables simultaneously (confounded — use cautiously)
```

---

## Agent v1 — Baseline (`bioai_agent_v1.py`)

**3 models · PMC full text · No system prompt**

Queries three general LLMs with full-text evidence but without clinical expert framing. Serves as the **baseline** for measuring what the biomedical system prompt contributes.

### Models

| # | Model | Size |
|---|---|---|
| 1 | `meta-llama/Llama-3.2-3B-Instruct` | 3B |
| 2 | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| 3 | `meta-llama/Llama-3.1-8B-Instruct` | 8B |

### Commands

| Command | Description |
|---|---|
| `/models` | List active models |
| `/pubmed` | Toggle evidence retrieval on/off |
| `/fulltext` | Toggle PMC full text vs abstract-only mode |
| `/evidence` | Show last retrieved evidence sources |
| `/history` | Show conversation history |
| `/clear` | Clear history |
| `/save` | Show CSV path |
| `/quit` | Exit |

---

## Agent v2 — System Prompt (`bioai_agent_biomedical.py`)

**5 models · Abstract only · Biomedical system prompt**

Queries five LLMs ranked by biomedical benchmark performance (USMLE, MedQA, PubMedQA) with an expert clinical persona injected on every query. Uses abstract-only evidence — serves as a comparison point for measuring full-text contribution.

### Models

| # | Model | Size | Benchmark Strength |
|---|---|---|---|
| 1 | `meta-llama/Llama-3.3-70B-Instruct` | 70B | Best reasoning |
| 2 | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Strong medical QA |
| 3 | `Qwen/Qwen2.5-7B-Instruct` | 7B | Biomedical benchmarks |
| 4 | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Lightweight, fast |
| 5 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Baseline |

### Additional Commands (beyond v1)

| Command | Description |
|---|---|
| `/enable <n>` | Enable model at runtime e.g. `/enable 1` |
| `/disable <n>` | Disable model at runtime e.g. `/disable 5` |
| `/score` | Show latency scoreboard |

---

## Agent v3 — Full Pipeline (`bioai_agent_v3.py`) ★ Recommended

**5 models · PMC full text · Biomedical system prompt**

The complete clinical research agent. Combines all capabilities — full-text evidence retrieval, biomedical expert framing, and 5-model comparison. Recommended for clinical decision making research.

### Models

Same as v2 (5 models ranked by biomedical benchmark performance).

### Additional Commands (beyond v2)

| Command | Description |
|---|---|
| `/fulltext` | Toggle PMC full text vs abstract-only |
| `/evidence` | Show last retrieved evidence sources |

### Session Stats on Exit

```
📈 Session Evidence Stats:
   Total articles retrieved: 9
   Full text (PMC):          4  (44%)
   Abstract only:            5
```

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

### 5. Run an agent
```bash
# v3 — recommended for clinical research
python bioai_agent_v3.py

# v1 — baseline comparison
python bioai_agent_v1.py

# v2 — system prompt, abstract only
python bioai_agent_biomedical.py
```

---

## Example Clinical Prompts

```
You > Is mechanical CPR superior to manual CPR in prehospital cardiac arrest?

You > Compare SGLT2 inhibitors and GLP-1 agonists in type 2 diabetes with heart failure

You > First-line treatment for community-acquired pneumonia in adults — current guidelines

You > What is the evidence for early goal-directed therapy in sepsis post-ProCESS trial?

You > DOAC vs warfarin in AF patients with chronic kidney disease — subgroup analysis
```

---

## Output & Logging

Every session automatically logs to CSV:

| Agent | CSV file |
|---|---|
| v1 | `~/bioai_results.csv` |
| v2 | `~/bioai_biomedical_results.csv` |
| v3 | `~/bioai_v3_results.csv` |

**CSV columns (v3):**
`timestamp · session_id · turn · prompt · model_alias · model_id · response · latency_s · response_words · pubmed_pmids · pmc_ids · fulltext_count · abstract_count`

### MLflow Tracking
```bash
mlflow ui
# Open http://localhost:5000
```

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Linux / macOS / Windows (WSL) |
| Python | 3.10+ |
| RAM | 4GB minimum |
| GPU | Not required |
| Internet | Required (API + PubMed calls) |
| HF Account | Free tier sufficient |

> Developed on: AMD Ryzen 5 5500 · 16GB RAM · AMD RX 560 · Ubuntu 24.04

---

## Limitations

- True biomedical fine-tuned models (BioMistral, Meditron, OpenBioLLM) require local GPU or paid cloud inference — not available on HF free serverless tier
- General models with biomedical system prompts + PubMed RAG are a scientifically justified practical alternative
- PMC full text available for ~40% of PubMed articles — remainder falls back to abstract
- HuggingFace free tier has rate limits — heavy usage may hit quotas
- Full-text retrieval adds ~3-8 seconds per article (PMC API calls)

---

## Roadmap

- [ ] Replicate API integration (BioMistral-7B — true biomedical fine-tuned model)
- [ ] Kaggle notebook version (30hr/week free GPU)
- [ ] Automated response quality scoring per model
- [ ] HTML side-by-side comparison report generator
- [ ] Meditron-7B (pending gated access approval)

---

## Repository Structure

```
bioai-agent/
├── bioai_agent_v1.py           # v1.2 — 3 models, full text, no system prompt (baseline)
├── bioai_agent_biomedical.py   # v2   — 5 models, system prompt, abstract only
├── bioai_agent_v3.py           # v3   — 5 models, system prompt, full text (recommended)
├── bioai_agent.py              # original v1.0 (reference)
├── test_biogpt.py              # quick HF API connectivity test
├── requirements.txt
├── .gitignore
└── README.md
```

---

## References

- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- [NCBI E-utilities (PubMed API)](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [PubMed Central Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Lewis et al. (2020) — Retrieval-Augmented Generation for NLP](https://arxiv.org/abs/2005.11401)
- [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)
- [Meditron-7B](https://huggingface.co/epfl-llm/meditron-7b)

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Concept & Domain Expertise: Dr. Abu Galib · Technical Implementation: Claude AI (Anthropic)*
