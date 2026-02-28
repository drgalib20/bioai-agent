"""
BioAI Agent v2 — Top 5 Biomedical LLM Comparison Agent
--------------------------------------------------------
Strategy:
  - True biomedical fine-tuned models are NOT available on HF serverless inference.
  - This agent uses the best AVAILABLE models with strong biomedical system prompts
    + PubMed grounding to maximise biomedical accuracy.
  - Models are ranked by biomedical benchmark performance (USMLE, MedQA, PubMedQA).

TOP 5 MODELS (by biomedical performance, all serverless-accessible):
  1. meta-llama/Llama-3.3-70B-Instruct   — best reasoning, top USMLE scores
  2. meta-llama/Llama-3.1-8B-Instruct    — strong medical QA, fast
  3. Qwen/Qwen2.5-7B-Instruct            — excellent biomedical benchmarks
  4. meta-llama/Llama-3.2-3B-Instruct    — lightweight, confirmed working
  5. mistralai/Mistral-7B-Instruct-v0.3  — solid medical reasoning baseline

NOTE on true biomedical models (BioMistral, Meditron, OpenBioLLM):
  These require local GPU or paid HF Inference Endpoints. Not on free serverless.
  Your PubMed grounding compensates by injecting real literature context.

Features:
  - Biomedical system prompt injected into every query
  - Multi-turn conversation history
  - PubMed search grounding (toggle on/off)
  - Per-model response comparison
  - CSV + MLflow logging
  - Model enable/disable at runtime
  - Response scoring (length, latency)

Usage:
  source /mnt/User/python-envs/bioai/bin/activate
  python ~/bioai_agent_biomedical.py
"""

import os
import csv
import time
import datetime
import requests
import json

from huggingface_hub import InferenceClient

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Top 5 models ranked by biomedical benchmark performance
MODELS = {
    "1_Llama3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct",   # Best - large
    "2_Llama3.1-8B":   "meta-llama/Llama-3.1-8B-Instruct",    # Strong - medium
    "3_Qwen2.5-7B":    "Qwen/Qwen2.5-7B-Instruct",            # Good - medium
    "4_Llama3.2-3B":   "meta-llama/Llama-3.2-3B-Instruct",    # Fast - small
    "5_Mistral-7B":    "mistralai/Mistral-7B-Instruct-v0.3",   # Baseline
}

# Biomedical system prompt — injected into every query
BIOMEDICAL_SYSTEM_PROMPT = """You are an expert biomedical AI assistant with deep knowledge in:
- Clinical medicine and diagnostics
- Pharmacology and drug interactions
- Molecular biology and genomics
- Medical literature and evidence-based medicine
- Pathophysiology and disease mechanisms

When answering:
1. Cite specific mechanisms, pathways, or guidelines where relevant
2. Distinguish between established evidence and emerging research
3. Use precise medical terminology
4. Note important clinical caveats or contraindications
5. Reference drug classes, dosing principles, or lab values when appropriate

You are assisting a medical researcher or clinician for educational and research purposes."""

MAX_TOKENS      = 500
LOG_CSV         = os.path.expanduser("~/bioai_biomedical_results.csv")
MLFLOW_EXP_NAME = "bioai_biomedical_agent"
PUBMED_MAX      = 3

# ══════════════════════════════════════════════════════════════════════════════
# PUBMED SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_pubmed(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        r = requests.get(
            f"{base}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results,
                    "retmode": "json", "sort": "relevance"},
            timeout=10
        )
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        r2 = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids),
                    "rettype": "abstract", "retmode": "xml"},
            timeout=15
        )

        import re
        titles    = re.findall(r"<ArticleTitle>(.*?)</ArticleTitle>", r2.text, re.S)
        abstracts = re.findall(r"<AbstractText.*?>(.*?)</AbstractText>",  r2.text, re.S)
        years     = re.findall(r"<PubDate>.*?<Year>(\d{4})</Year>",      r2.text, re.S)

        results = []
        for i, pmid in enumerate(ids):
            results.append({
                "pmid":     pmid,
                "title":    titles[i]    if i < len(titles)    else "N/A",
                "abstract": abstracts[i] if i < len(abstracts) else "No abstract",
                "year":     years[i]     if i < len(years)     else "N/A",
            })
        return results
    except Exception as e:
        print(f"  [PubMed error: {e}]")
        return []


def format_pubmed_context(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = ["\n=== PubMed Evidence ==="]
    for a in articles:
        lines.append(f"[PMID {a['pmid']} ({a['year']})]: {a['title']}")
        lines.append(f"  Abstract: {a['abstract'][:400]}...")
    lines.append("======================\n")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "turn", "prompt",
                "model_alias", "model_id", "response",
                "latency_s", "response_words", "pubmed_pmids"
            ])

def log_csv(session_id, turn, prompt, alias, model_id, response, latency, pmids):
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            session_id, turn, prompt,
            alias, model_id,
            response.replace("\n", " "),
            round(latency, 2),
            len(response.split()),
            "|".join(pmids)
        ])

def init_mlflow():
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_experiment(MLFLOW_EXP_NAME)
        run = mlflow.start_run()
        mlflow.log_param("models", list(MODELS.keys()))
        mlflow.log_param("pubmed_max", PUBMED_MAX)
        mlflow.log_param("max_tokens", MAX_TOKENS)
        print(f"  [MLflow] Run: {run.info.run_id}")
        return run
    except Exception as e:
        print(f"  [MLflow unavailable: {e}]")
        return None

def log_mlflow(turn, alias, latency, word_count):
    if not MLFLOW_AVAILABLE:
        return
    try:
        safe = alias.replace("/", "_")
        mlflow.log_metrics({
            f"{safe}_latency_s":    round(latency, 2),
            f"{safe}_word_count":   word_count,
        }, step=turn)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# MODEL QUERY
# ══════════════════════════════════════════════════════════════════════════════

def query_model(client, model_id, history, prompt, pubmed_context):
    messages = [{"role": "system", "content": BIOMEDICAL_SYSTEM_PROMPT}]
    messages += history

    user_content = prompt
    if pubmed_context:
        user_content = (
            pubmed_context +
            "\nUsing the PubMed evidence above where relevant, answer this question:\n" +
            prompt
        )
    messages.append({"role": "user", "content": user_content})

    try:
        t0 = time.time()
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=MAX_TOKENS,
        )
        latency = time.time() - t0
        text = response.choices[0].message.content.strip()
        return text, latency
    except Exception as e:
        return f"[ERROR: {e}]", 0.0

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

W = 72

def banner():
    print(f"\n{'═'*W}")
    print("  🧬  BioAI Agent v2 — Top 5 Biomedical LLM Comparison")
    print(f"{'═'*W}")
    print("  Commands:")
    print("    /models          — list models + enable/disable")
    print("    /enable <num>    — enable model by number (e.g. /enable 1)")
    print("    /disable <num>   — disable model by number")
    print("    /pubmed          — toggle PubMed grounding")
    print("    /history         — show conversation history")
    print("    /clear           — clear history")
    print("    /score           — show latency scoreboard")
    print("    /save            — show CSV path")
    print("    /quit            — exit")
    print(f"{'═'*W}\n")

def print_models(active_models):
    print(f"\n  {'#':<4} {'Alias':<20} {'Model ID':<45} {'Status'}")
    print(f"  {'─'*70}")
    for alias, model_id in MODELS.items():
        num = alias.split("_")[0]
        status = "✅ ON " if alias in active_models else "❌ OFF"
        print(f"  {num:<4} {alias:<20} {model_id:<45} {status}")
    print()

def print_response(alias, text, latency, word_count):
    print(f"\n  ┌─ {alias} │ {latency:.1f}s │ {word_count} words {'─'*(W-len(alias)-20)}")
    for line in text.split("\n"):
        if line.strip():
            # wrap long lines
            while len(line) > W - 6:
                print(f"  │  {line[:W-6]}")
                line = "    " + line[W-6:]
            print(f"  │  {line}")
    print(f"  └{'─'*(W-2)}")

def print_scoreboard(scores):
    if not scores:
        return
    print(f"\n  📊 Latency Scoreboard (this session avg):")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["avg"])
    for alias, s in sorted_scores:
        bar = "█" * int(s["avg"] * 2)
        print(f"  {alias:<22} {s['avg']:>5.1f}s avg  {bar}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
        return

    client       = InferenceClient(token=token)
    session_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history      = []
    turn         = 0
    use_pubmed   = True
    active_models = dict(MODELS)   # start with all enabled
    latency_log  = {alias: {"total": 0, "count": 0, "avg": 0}
                    for alias in MODELS}

    init_csv()
    mlflow_run = init_mlflow()
    banner()
    print_models(active_models)

    print(f"  Biomedical system prompt: ✅ ACTIVE")
    print(f"  PubMed grounding:         {'✅ ON' if use_pubmed else '❌ OFF'}")
    print(f"  Results CSV:              {LOG_CSV}")
    print(f"\n  ⚠️  Note: These are general LLMs with biomedical system prompts.")
    print(f"  True biomedical models (BioMistral, Meditron) require local GPU.")
    print(f"  PubMed grounding compensates with real literature context.\n")

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────────────
            cmd = user_input.lower()

            if cmd == "/quit":
                break

            elif cmd == "/models":
                print_models(active_models)
                continue

            elif cmd.startswith("/enable "):
                num = cmd.split()[1]
                for alias in MODELS:
                    if alias.startswith(num + "_"):
                        active_models[alias] = MODELS[alias]
                        print(f"  ✅ Enabled: {alias}")
                continue

            elif cmd.startswith("/disable "):
                num = cmd.split()[1]
                for alias in list(active_models.keys()):
                    if alias.startswith(num + "_"):
                        del active_models[alias]
                        print(f"  ❌ Disabled: {alias}")
                continue

            elif cmd == "/pubmed":
                use_pubmed = not use_pubmed
                print(f"  PubMed grounding: {'✅ ON' if use_pubmed else '❌ OFF'}")
                continue

            elif cmd == "/history":
                if not history:
                    print("  No history yet.")
                for msg in history:
                    role = msg["role"].upper()
                    print(f"  [{role}] {msg['content'][:100]}...")
                continue

            elif cmd == "/clear":
                history.clear()
                turn = 0
                print("  ✅ History cleared.")
                continue

            elif cmd == "/score":
                print_scoreboard(latency_log)
                continue

            elif cmd == "/save":
                print(f"  📄 CSV: {LOG_CSV}")
                continue

            # ── PubMed ────────────────────────────────────────────────────
            pubmed_articles = []
            pubmed_context  = ""
            if use_pubmed:
                print(f"\n  🔍 PubMed: '{user_input[:60]}'...")
                pubmed_articles = search_pubmed(user_input)
                if pubmed_articles:
                    pubmed_context = format_pubmed_context(pubmed_articles)
                    for a in pubmed_articles:
                        print(f"    [{a['year']}] PMID {a['pmid']}: {a['title'][:65]}")
                else:
                    print("  No PubMed results.")

            pmids = [a["pmid"] for a in pubmed_articles]
            turn += 1
            print(f"\n{'─'*W}")
            print(f"  Turn {turn} | {len(active_models)} model(s) | "
                  f"PubMed {'ON' if use_pubmed else 'OFF'}")
            print(f"{'─'*W}")

            # ── Query each active model ───────────────────────────────────
            responses = {}
            for alias, model_id in active_models.items():
                print(f"  ⏳ {alias}...", end="", flush=True)
                text, latency = query_model(
                    client, model_id, history, user_input, pubmed_context
                )
                word_count = len(text.split())
                responses[alias] = (text, latency, word_count)

                # update latency tracker
                latency_log[alias]["total"] += latency
                latency_log[alias]["count"] += 1
                latency_log[alias]["avg"] = (
                    latency_log[alias]["total"] / latency_log[alias]["count"]
                )

                log_csv(session_id, turn, user_input,
                        alias, model_id, text, latency, pmids)
                log_mlflow(turn, alias, latency, word_count)
                print(f" ✅ {latency:.1f}s")

            # ── Print responses ───────────────────────────────────────────
            for alias, (text, latency, word_count) in responses.items():
                print_response(alias, text, latency, word_count)

            # ── Update history ────────────────────────────────────────────
            history.append({"role": "user", "content": user_input})
            # use best model's response for history continuity
            best = list(responses.items())[0]
            history.append({"role": "assistant", "content": best[1][0]})

            print(f"\n  📄 Logged → {LOG_CSV}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()
        if any(v["count"] > 0 for v in latency_log.values()):
            print_scoreboard(latency_log)
        print(f"\n  Session ended. Results: {LOG_CSV}")
        print("  Goodbye! 🧬\n")


if __name__ == "__main__":
    main()
