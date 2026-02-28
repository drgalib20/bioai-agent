"""
BioAI Agent — Biomedical LLM Testing & Comparison Agent
--------------------------------------------------------
Features:
  - Multi-turn conversation history
  - Send prompts to multiple HF-hosted LLMs simultaneously
  - PubMed search to ground answers in literature
  - Auto-log all results to CSV and MLflow
  - Terminal chat interface

Usage:
  python bioai_agent.py
"""

import os
import csv
import time
import json
import datetime
import requests
from huggingface_hub import InferenceClient

# ── Optional MLflow (graceful fallback if not configured) ──────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit as needed
# ══════════════════════════════════════════════════════════════════════════════

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

MAX_TOKENS       = 400
LOG_CSV          = os.path.expanduser("~/bioai_results.csv")
MLFLOW_EXP_NAME  = "bioai_agent"
PUBMED_MAX       = 3       # number of PubMed abstracts to retrieve

# ══════════════════════════════════════════════════════════════════════════════
# PUBMED SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_pubmed(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    """Search PubMed and return a list of {title, abstract, pmid}."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        # Step 1: get PMIDs
        search_url = f"{base}/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_results,
                  "retmode": "json", "sort": "relevance"}
        r = requests.get(search_url, params=params, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2: fetch summaries
        fetch_url = f"{base}/efetch.fcgi"
        params2 = {"db": "pubmed", "id": ",".join(ids),
                   "rettype": "abstract", "retmode": "xml"}
        r2 = requests.get(fetch_url, params=params2, timeout=15)

        # Simple XML parse (no lxml needed)
        import re
        titles    = re.findall(r"<ArticleTitle>(.*?)</ArticleTitle>", r2.text, re.S)
        abstracts = re.findall(r"<AbstractText.*?>(.*?)</AbstractText>",  r2.text, re.S)

        results = []
        for i, pmid in enumerate(ids):
            results.append({
                "pmid":     pmid,
                "title":    titles[i]    if i < len(titles)    else "N/A",
                "abstract": abstracts[i] if i < len(abstracts) else "No abstract available",
            })
        return results

    except Exception as e:
        print(f"  [PubMed error: {e}]")
        return []


def format_pubmed_context(articles: list[dict]) -> str:
    if not articles:
        return ""
    lines = ["\n--- PubMed Context ---"]
    for a in articles:
        lines.append(f"PMID {a['pmid']}: {a['title']}")
        lines.append(f"  {a['abstract'][:300]}...")
    lines.append("----------------------\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "session_id", "turn",
                             "prompt", "model", "response", "latency_s",
                             "pubmed_pmids"])


def log_to_csv(session_id, turn, prompt, model, response, latency, pmids):
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(),
            session_id, turn, prompt, model,
            response.replace("\n", " "), round(latency, 2),
            "|".join(pmids)
        ])


# ══════════════════════════════════════════════════════════════════════════════
# MLFLOW LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_mlflow():
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_experiment(MLFLOW_EXP_NAME)
        run = mlflow.start_run()
        print(f"  [MLflow] Run ID: {run.info.run_id}")
        return run
    except Exception as e:
        print(f"  [MLflow unavailable: {e}]")
        return None


def log_to_mlflow(turn, model, latency, prompt_len, response_len):
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_metrics({
            f"{model.split('/')[-1]}_latency":      latency,
            f"{model.split('/')[-1]}_response_len": response_len,
        }, step=turn)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# QUERY A SINGLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def query_model(client: InferenceClient, model: str,
                history: list[dict], prompt: str,
                pubmed_context: str) -> tuple[str, float]:
    """Send prompt + history to a model, return (response_text, latency)."""

    # Build message list: history + optional pubmed context + current prompt
    messages = list(history)
    user_content = prompt
    if pubmed_context:
        user_content = pubmed_context + "\nUsing the above context, answer: " + prompt
    messages.append({"role": "user", "content": user_content})

    try:
        t0 = time.time()
        response = client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=MAX_TOKENS,
        )
        latency = time.time() - t0
        text = response.choices[0].message.content.strip()
        return text, latency
    except Exception as e:
        return f"[ERROR: {e}]", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DIVIDER  = "═" * 70
DIVIDER2 = "─" * 70

def print_banner():
    print(f"\n{DIVIDER}")
    print("  🧬  BioAI Agent — Biomedical LLM Comparison Terminal")
    print(f"{DIVIDER}")
    print("  Commands:")
    print("    /models   — list active models")
    print("    /pubmed   — toggle PubMed grounding (on/off)")
    print("    /history  — show conversation history")
    print("    /clear    — clear conversation history")
    print("    /save     — show path to results CSV")
    print("    /quit     — exit")
    print(f"{DIVIDER}\n")


def print_response(model: str, text: str, latency: float):
    short = model.split("/")[-1]
    print(f"\n  ┌─ {short} ({latency:.1f}s) {'─'*(55-len(short))}")
    for line in text.split("\n"):
        print(f"  │  {line}")
    print(f"  └{'─'*67}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Run: export HF_TOKEN='hf_...'")
        return

    client     = InferenceClient(token=token)
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history    = []          # shared conversation history
    turn       = 0
    use_pubmed = True
    active_models = list(MODELS)

    init_csv()
    mlflow_run = init_mlflow()
    print_banner()

    print(f"  Active models ({len(active_models)}):")
    for m in active_models:
        print(f"    • {m}")
    print(f"\n  PubMed grounding: {'ON' if use_pubmed else 'OFF'}")
    print(f"  Results logged to: {LOG_CSV}\n")

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────────────
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/models":
                print("\n  Active models:")
                for m in active_models:
                    print(f"    • {m}")
                continue
            elif user_input.lower() == "/pubmed":
                use_pubmed = not use_pubmed
                print(f"  PubMed grounding: {'ON' if use_pubmed else 'OFF'}")
                continue
            elif user_input.lower() == "/history":
                if not history:
                    print("  No history yet.")
                for msg in history:
                    print(f"  [{msg['role'].upper()}] {msg['content'][:120]}...")
                continue
            elif user_input.lower() == "/clear":
                history.clear()
                turn = 0
                print("  Conversation history cleared.")
                continue
            elif user_input.lower() == "/save":
                print(f"  CSV saved at: {LOG_CSV}")
                continue

            # ── PubMed search ─────────────────────────────────────────────
            pubmed_articles = []
            pubmed_context  = ""
            if use_pubmed:
                print(f"\n  🔍 Searching PubMed for: '{user_input[:60]}'...")
                pubmed_articles = search_pubmed(user_input)
                if pubmed_articles:
                    pubmed_context = format_pubmed_context(pubmed_articles)
                    print(f"  Found {len(pubmed_articles)} article(s):")
                    for a in pubmed_articles:
                        print(f"    PMID {a['pmid']}: {a['title'][:70]}")
                else:
                    print("  No PubMed results found.")

            pmids = [a["pmid"] for a in pubmed_articles]
            turn += 1
            print(f"\n{DIVIDER2}")
            print(f"  Turn {turn} — querying {len(active_models)} model(s)...")
            print(DIVIDER2)

            # ── Query each model ──────────────────────────────────────────
            responses = {}
            for model in active_models:
                short = model.split("/")[-1]
                print(f"  ⏳ {short}...", end="", flush=True)
                text, latency = query_model(client, model, history,
                                            user_input, pubmed_context)
                responses[model] = (text, latency)
                print(f" done ({latency:.1f}s)")

                log_to_csv(session_id, turn, user_input,
                           model, text, latency, pmids)
                log_to_mlflow(turn, model, latency,
                              len(user_input), len(text))

            # ── Print all responses ───────────────────────────────────────
            for model, (text, latency) in responses.items():
                print_response(model, text, latency)

            # ── Update shared history with first model's response ─────────
            history.append({"role": "user",      "content": user_input})
            first_response = list(responses.values())[0][0]
            history.append({"role": "assistant", "content": first_response})

            print(f"\n  📄 Logged to: {LOG_CSV}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()
        print(f"\n  Session ended. Results saved to: {LOG_CSV}")
        print("  Goodbye! 🧬\n")


if __name__ == "__main__":
    main()
