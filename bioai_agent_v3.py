"""
BioAI Agent v3 — Full-Text PMC + PubMed Evidence Grounding
------------------------------------------------------------
Upgrade from v2:
  - Retrieves FREE FULL TEXT from PubMed Central (PMC) where available
  - Falls back gracefully to abstract if full text not in PMC
  - Shows evidence source clearly: [FULL TEXT] vs [ABSTRACT]
  - Extracts key sections: Introduction, Methods, Results, Discussion
  - Truncates intelligently to fit LLM context window
  - Tracks full_text vs abstract ratio per session

Evidence retrieval pipeline (per article):
  1. PubMed esearch  → get PMIDs
  2. PubMed efetch   → get abstract + check for PMC ID
  3. PMC efetch      → attempt full text XML if PMCID exists
  4. Section parser  → extract Results + Discussion (most clinically relevant)
  5. Fallback        → use abstract if no PMC full text available

TOP 5 MODELS (by biomedical benchmark performance):
  1. meta-llama/Llama-3.3-70B-Instruct   — best reasoning, top USMLE scores
  2. meta-llama/Llama-3.1-8B-Instruct    — strong medical QA, fast
  3. Qwen/Qwen2.5-7B-Instruct            — excellent biomedical benchmarks
  4. meta-llama/Llama-3.2-3B-Instruct    — lightweight, confirmed working
  5. mistralai/Mistral-7B-Instruct-v0.3  — solid medical reasoning baseline

Usage:
  source /mnt/User/python-envs/bioai/bin/activate
  python ~/bioai_agent_v3.py
"""

import os
import re
import csv
import time
import datetime
import requests

from huggingface_hub import InferenceClient

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "1_Llama3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "2_Llama3.1-8B":  "meta-llama/Llama-3.1-8B-Instruct",
    "3_Qwen2.5-7B":   "Qwen/Qwen2.5-7B-Instruct",
    "4_Llama3.2-3B":  "meta-llama/Llama-3.2-3B-Instruct",
    "5_Mistral-7B":   "mistralai/Mistral-7B-Instruct-v0.3",
}

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
4. Note important clinical caveats, contraindications, and subgroup effects
5. Reference drug classes, dosing principles, or lab values when appropriate
6. When full-text evidence is provided, leverage detailed methodology,
   results, and discussion sections — not just the abstract summary

You are assisting a medical researcher or clinician for educational and research purposes."""

MAX_TOKENS          = 600       # increased for richer full-text context
LOG_CSV             = os.path.expanduser("~/bioai_v3_results.csv")
MLFLOW_EXP_NAME     = "bioai_agent_v3"
PUBMED_MAX          = 3         # articles per query
FULLTEXT_MAX_CHARS  = 3000      # max chars of full text to inject per article
ABSTRACT_MAX_CHARS  = 500       # max chars of abstract to inject per article

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ══════════════════════════════════════════════════════════════════════════════
# PMC FULL TEXT RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def get_pmc_id(pmid: str) -> str | None:
    """
    Convert a PubMed PMID to a PMC ID using the ID converter API.
    Returns 'PMC1234567' string or None if not in PMC.
    """
    try:
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        r = requests.get(url, params={
            "ids": pmid,
            "format": "json",
            "tool": "bioai_agent",
            "email": "bioai@research.local"
        }, timeout=10)
        records = r.json().get("records", [])
        if records and "pmcid" in records[0]:
            return records[0]["pmcid"]   # e.g. "PMC7654321"
        return None
    except Exception:
        return None


def fetch_pmc_fulltext(pmcid: str) -> dict:
    """
    Fetch full text XML from PubMed Central.
    Returns dict with extracted sections.
    """
    try:
        # Strip 'PMC' prefix for efetch
        pmc_num = pmcid.replace("PMC", "")
        r = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db":      "pmc",
                "id":      pmc_num,
                "rettype": "full",
                "retmode": "xml",
            },
            timeout=20
        )

        if r.status_code != 200 or len(r.text) < 500:
            return {}

        xml = r.text

        # ── Extract sections by title ──────────────────────────────────────
        sections = {}

        # Extract all <sec> blocks with their titles
        sec_blocks = re.findall(
            r'<sec[^>]*>.*?<title>(.*?)</title>(.*?)</sec>',
            xml, re.S | re.I
        )

        for title, content in sec_blocks:
            # Clean XML tags from content
            clean = re.sub(r'<[^>]+>', ' ', content)
            clean = re.sub(r'\s+', ' ', clean).strip()
            title_clean = re.sub(r'<[^>]+>', '', title).strip().upper()
            if clean and len(clean) > 50:
                sections[title_clean] = clean

        # ── Also extract abstract if not already retrieved ─────────────────
        abstract_match = re.search(
            r'<abstract[^>]*>(.*?)</abstract>', xml, re.S | re.I
        )
        if abstract_match:
            abstract_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
            sections['ABSTRACT'] = re.sub(r'\s+', ' ', abstract_text).strip()

        return sections

    except Exception as e:
        return {}


def extract_key_sections(sections: dict, max_chars: int = FULLTEXT_MAX_CHARS) -> str:
    """
    From all extracted sections, prioritise the most clinically
    relevant ones: Results, Discussion, Conclusions, then others.
    Truncates to max_chars total.
    """
    if not sections:
        return ""

    # Priority order for clinical relevance
    priority = [
        'RESULTS', 'DISCUSSION', 'CONCLUSIONS', 'CONCLUSION',
        'FINDINGS', 'CLINICAL IMPLICATIONS', 'ABSTRACT',
        'INTRODUCTION', 'BACKGROUND', 'METHODS', 'MATERIALS AND METHODS'
    ]

    ordered = []

    # Add priority sections first
    for key in priority:
        for sec_title, content in sections.items():
            if key in sec_title and content not in ordered:
                ordered.append((sec_title, content))
                break

    # Add any remaining sections not already included
    included_contents = {c for _, c in ordered}
    for sec_title, content in sections.items():
        if content not in included_contents:
            ordered.append((sec_title, content))
            included_contents.add(content)

    # Build output up to max_chars
    output_parts = []
    total = 0
    for title, content in ordered:
        if total >= max_chars:
            break
        remaining = max_chars - total
        snippet = content[:remaining]
        output_parts.append(f"[{title}]\n{snippet}")
        total += len(snippet)

    return "\n\n".join(output_parts)


# ══════════════════════════════════════════════════════════════════════════════
# PUBMED SEARCH — ENHANCED WITH FULL TEXT ATTEMPT
# ══════════════════════════════════════════════════════════════════════════════

def search_pubmed_fulltext(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    """
    Search PubMed for articles matching query.
    For each article, attempt to retrieve PMC full text.
    Falls back to abstract if full text unavailable.

    Returns list of dicts with:
      pmid, title, year, abstract, full_text, pmcid, source
      where source = 'FULL TEXT' | 'ABSTRACT'
    """
    try:
        # ── Step 1: Get PMIDs ──────────────────────────────────────────────
        r = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={
                "db":      "pubmed",
                "term":    query,
                "retmax":  max_results,
                "retmode": "json",
                "sort":    "relevance"
            },
            timeout=10
        )
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # ── Step 2: Fetch PubMed abstracts + metadata ──────────────────────
        r2 = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(ids),
                "rettype": "abstract",
                "retmode": "xml"
            },
            timeout=15
        )

        titles    = re.findall(r"<ArticleTitle>(.*?)</ArticleTitle>", r2.text, re.S)
        abstracts = re.findall(r"<AbstractText.*?>(.*?)</AbstractText>", r2.text, re.S)
        years     = re.findall(r"<PubDate>.*?<Year>(\d{4})</Year>", r2.text, re.S)
        dois      = re.findall(r'<ArticleId IdType="doi">(.*?)</ArticleId>', r2.text, re.S)

        results = []

        for i, pmid in enumerate(ids):
            article = {
                "pmid":      pmid,
                "title":     titles[i]    if i < len(titles)    else "N/A",
                "abstract":  abstracts[i] if i < len(abstracts) else "No abstract available",
                "year":      years[i]     if i < len(years)     else "N/A",
                "doi":       dois[i]      if i < len(dois)       else "N/A",
                "pmcid":     None,
                "full_text": None,
                "source":    "ABSTRACT",   # default
            }

            # ── Step 3: Try to get PMC full text ──────────────────────────
            print(f"    🔎 Checking PMC for PMID {pmid}...", end="", flush=True)
            pmcid = get_pmc_id(pmid)

            if pmcid:
                article["pmcid"] = pmcid
                print(f" {pmcid} found, fetching...", end="", flush=True)
                sections = fetch_pmc_fulltext(pmcid)

                if sections:
                    full_text = extract_key_sections(sections)
                    if full_text and len(full_text) > 200:
                        article["full_text"] = full_text
                        article["source"]    = "FULL TEXT"
                        print(f" ✅ {len(full_text)} chars")
                    else:
                        print(f" ⚠️ Too short, using abstract")
                else:
                    print(f" ⚠️ Parse failed, using abstract")
            else:
                print(f" ❌ Not in PMC, using abstract")

            results.append(article)

        return results

    except Exception as e:
        print(f"\n  [Evidence retrieval error: {e}]")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT EVIDENCE CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def format_evidence_context(articles: list[dict]) -> str:
    """
    Format retrieved articles into LLM context.
    Full-text articles get richer sections;
    abstract-only articles get truncated abstract.
    """
    if not articles:
        return ""

    lines = ["\n=== Evidence from PubMed/PMC ==="]

    for a in articles:
        source_tag = f"[{a['source']}]"
        pmcid_tag  = f" | {a['pmcid']}" if a['pmcid'] else ""
        lines.append(
            f"\n{source_tag} PMID {a['pmid']}{pmcid_tag} ({a['year']})"
        )
        lines.append(f"Title: {a['title']}")

        if a['source'] == "FULL TEXT" and a['full_text']:
            lines.append(f"Full Text (key sections):\n{a['full_text']}")
        else:
            abstract = a['abstract'][:ABSTRACT_MAX_CHARS]
            lines.append(f"Abstract: {abstract}...")

    lines.append("\n================================\n")
    return "\n".join(lines)


def evidence_summary(articles: list[dict]) -> str:
    """One-line summary of evidence sources for terminal display."""
    full  = sum(1 for a in articles if a['source'] == 'FULL TEXT')
    abstr = sum(1 for a in articles if a['source'] == 'ABSTRACT')
    parts = []
    if full:
        parts.append(f"📄 {full} full text")
    if abstr:
        parts.append(f"📋 {abstr} abstract only")
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "turn", "prompt",
                "model_alias", "model_id", "response",
                "latency_s", "response_words",
                "pubmed_pmids", "pmc_ids", "fulltext_count", "abstract_count"
            ])


def log_csv(session_id, turn, prompt, alias, model_id,
            response, latency, articles):
    pmids      = [a["pmid"]  for a in articles]
    pmc_ids    = [a["pmcid"] for a in articles if a["pmcid"]]
    ft_count   = sum(1 for a in articles if a["source"] == "FULL TEXT")
    abs_count  = sum(1 for a in articles if a["source"] == "ABSTRACT")

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            session_id, turn, prompt,
            alias, model_id,
            response.replace("\n", " "),
            round(latency, 2),
            len(response.split()),
            "|".join(pmids),
            "|".join(pmc_ids),
            ft_count,
            abs_count,
        ])


def init_mlflow():
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_experiment(MLFLOW_EXP_NAME)
        run = mlflow.start_run()
        mlflow.log_param("models",      list(MODELS.keys()))
        mlflow.log_param("pubmed_max",  PUBMED_MAX)
        mlflow.log_param("max_tokens",  MAX_TOKENS)
        mlflow.log_param("fulltext_max_chars", FULLTEXT_MAX_CHARS)
        print(f"  [MLflow] Run: {run.info.run_id}")
        return run
    except Exception as e:
        print(f"  [MLflow unavailable: {e}]")
        return None


def log_mlflow(turn, alias, latency, word_count, ft_count):
    if not MLFLOW_AVAILABLE:
        return
    try:
        safe = alias.replace("/", "_")
        mlflow.log_metrics({
            f"{safe}_latency_s":    round(latency, 2),
            f"{safe}_word_count":   word_count,
            f"fulltext_retrieved":  ft_count,
        }, step=turn)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# MODEL QUERY
# ══════════════════════════════════════════════════════════════════════════════

def query_model(client, model_id, history, prompt, evidence_context):
    messages = [{"role": "system", "content": BIOMEDICAL_SYSTEM_PROMPT}]
    messages += history

    user_content = prompt
    if evidence_context:
        user_content = (
            evidence_context +
            "\nUsing the evidence above (full text where available), "
            "answer this clinical/biomedical question in detail:\n" +
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
    print("  🧬  BioAI Agent v3 — Full-Text PMC + PubMed Evidence")
    print(f"{'═'*W}")
    print("  Evidence: Full text from PMC where available, abstract fallback")
    print("  Commands:")
    print("    /models          — list models")
    print("    /enable <num>    — enable model  (e.g. /enable 1)")
    print("    /disable <num>   — disable model (e.g. /disable 5)")
    print("    /pubmed          — toggle evidence retrieval on/off")
    print("    /fulltext        — toggle full-text mode (full text vs abstract only)")
    print("    /history         — show conversation history")
    print("    /clear           — clear history")
    print("    /score           — show latency scoreboard")
    print("    /evidence        — show last retrieved evidence sources")
    print("    /save            — show CSV path")
    print("    /quit            — exit")
    print(f"{'═'*W}\n")


def print_models(active_models):
    print(f"\n  {'#':<4} {'Alias':<20} {'Model ID':<45} {'Status'}")
    print(f"  {'─'*70}")
    for alias, model_id in MODELS.items():
        num    = alias.split("_")[0]
        status = "✅ ON " if alias in active_models else "❌ OFF"
        print(f"  {num:<4} {alias:<20} {model_id:<45} {status}")
    print()


def print_response(alias, text, latency, word_count):
    header = f"  ┌─ {alias} │ {latency:.1f}s │ {word_count} words "
    print(f"\n{header}{'─'*max(0, W - len(header) + 2)}")
    for line in text.split("\n"):
        if line.strip():
            while len(line) > W - 6:
                print(f"  │  {line[:W-6]}")
                line = "    " + line[W-6:]
            print(f"  │  {line}")
    print(f"  └{'─'*(W-2)}")


def print_scoreboard(scores):
    if not scores:
        return
    print(f"\n  📊 Latency Scoreboard:")
    for alias, s in sorted(scores.items(), key=lambda x: x[1]["avg"]):
        if s["count"] > 0:
            bar = "█" * int(s["avg"] * 2)
            print(f"  {alias:<22} {s['avg']:>5.1f}s avg  {bar}")
    print()


def print_evidence_list(articles):
    if not articles:
        print("  No evidence retrieved yet.")
        return
    print(f"\n  📚 Evidence retrieved ({len(articles)} articles):")
    for a in articles:
        icon = "📄" if a['source'] == 'FULL TEXT' else "📋"
        pmcid = f" | {a['pmcid']}" if a['pmcid'] else ""
        print(f"  {icon} [{a['year']}] PMID {a['pmid']}{pmcid}")
        print(f"     {a['source']}: {a['title'][:65]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATS
# ══════════════════════════════════════════════════════════════════════════════

def print_session_stats(stats):
    print(f"\n  📈 Session Evidence Stats:")
    print(f"     Total articles retrieved: {stats['total_articles']}")
    print(f"     Full text (PMC):          {stats['fulltext']} articles")
    print(f"     Abstract only:            {stats['abstract_only']} articles")
    if stats['total_articles'] > 0:
        pct = (stats['fulltext'] / stats['total_articles']) * 100
        print(f"     Full-text rate:           {pct:.0f}%")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Run: export HF_TOKEN='hf_...'")
        return

    client        = InferenceClient(token=token)
    session_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    history       = []
    turn          = 0
    use_pubmed    = True
    use_fulltext  = True        # NEW: toggle full-text mode
    active_models = dict(MODELS)
    latency_log   = {alias: {"total": 0, "count": 0, "avg": 0}
                     for alias in MODELS}

    # Session-level evidence stats
    session_stats = {"total_articles": 0, "fulltext": 0, "abstract_only": 0}

    # Keep last retrieved articles for /evidence command
    last_articles = []

    init_csv()
    mlflow_run = init_mlflow()
    banner()
    print_models(active_models)

    print(f"  Biomedical system prompt: ✅ ACTIVE")
    print(f"  Evidence retrieval:       {'✅ ON' if use_pubmed else '❌ OFF'}")
    print(f"  Full-text mode (PMC):     {'✅ ON' if use_fulltext else '❌ OFF (abstract only)'}")
    print(f"  Results CSV:              {LOG_CSV}")
    print(f"\n  ⚠️  Full-text retrieval adds ~3-8s per article (PMC API calls)")
    print(f"  Use /fulltext to toggle to abstract-only mode for faster queries.\n")

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            # ── Commands ──────────────────────────────────────────────────
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
                print(f"  Evidence retrieval: {'✅ ON' if use_pubmed else '❌ OFF'}")
                continue

            elif cmd == "/fulltext":
                use_fulltext = not use_fulltext
                mode = "✅ FULL TEXT (PMC)" if use_fulltext else "📋 ABSTRACT ONLY (faster)"
                print(f"  Evidence mode: {mode}")
                continue

            elif cmd == "/history":
                if not history:
                    print("  No history yet.")
                for msg in history:
                    print(f"  [{msg['role'].upper()}] {msg['content'][:100]}...")
                continue

            elif cmd == "/clear":
                history.clear()
                turn = 0
                print("  ✅ History cleared.")
                continue

            elif cmd == "/score":
                print_scoreboard(latency_log)
                continue

            elif cmd == "/evidence":
                print_evidence_list(last_articles)
                continue

            elif cmd == "/save":
                print(f"  📄 CSV: {LOG_CSV}")
                continue

            # ── Evidence Retrieval ────────────────────────────────────────
            articles       = []
            evidence_context = ""

            if use_pubmed:
                print(f"\n  🔍 Searching: '{user_input[:60]}'...")

                if use_fulltext:
                    print(f"  📄 Full-text mode — checking PMC for each article...")
                    articles = search_pubmed_fulltext(user_input)
                else:
                    # Abstract-only fast path (same as v2)
                    articles = search_pubmed_abstractonly(user_input)

                if articles:
                    last_articles = articles
                    evidence_context = format_evidence_context(articles)

                    # Update session stats
                    session_stats["total_articles"] += len(articles)
                    for a in articles:
                        if a["source"] == "FULL TEXT":
                            session_stats["fulltext"] += 1
                        else:
                            session_stats["abstract_only"] += 1

                    # Print summary
                    print(f"\n  📚 Evidence retrieved: {evidence_summary(articles)}")
                    for a in articles:
                        icon = "📄" if a['source'] == 'FULL TEXT' else "📋"
                        pmcid = f" ({a['pmcid']})" if a['pmcid'] else ""
                        print(f"    {icon} [{a['year']}] PMID {a['pmid']}{pmcid}: {a['title'][:55]}")
                else:
                    print("  ⚠️ No evidence found.")

            ft_count = sum(1 for a in articles if a["source"] == "FULL TEXT")
            turn += 1
            print(f"\n{'─'*W}")
            print(f"  Turn {turn} | {len(active_models)} model(s) | "
                  f"Evidence {'ON' if use_pubmed else 'OFF'} | "
                  f"{'Full text' if use_fulltext else 'Abstract'}")
            print(f"{'─'*W}")

            # ── Query each active model ───────────────────────────────────
            responses = {}
            for alias, model_id in active_models.items():
                print(f"  ⏳ {alias}...", end="", flush=True)
                text, latency = query_model(
                    client, model_id, history, user_input, evidence_context
                )
                word_count = len(text.split())
                responses[alias] = (text, latency, word_count)

                latency_log[alias]["total"] += latency
                latency_log[alias]["count"] += 1
                latency_log[alias]["avg"] = (
                    latency_log[alias]["total"] / latency_log[alias]["count"]
                )

                log_csv(session_id, turn, user_input,
                        alias, model_id, text, latency, articles)
                log_mlflow(turn, alias, latency, word_count, ft_count)
                print(f" ✅ {latency:.1f}s")

            # ── Print responses ───────────────────────────────────────────
            for alias, (text, latency, word_count) in responses.items():
                print_response(alias, text, latency, word_count)

            # ── Update history ────────────────────────────────────────────
            history.append({"role": "user", "content": user_input})
            best = list(responses.items())[0]
            history.append({"role": "assistant", "content": best[1][0]})

            print(f"\n  📄 Logged → {LOG_CSV}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()
        if any(v["count"] > 0 for v in latency_log.values()):
            print_scoreboard(latency_log)
        print_session_stats(session_stats)
        print(f"\n  Session ended. Results: {LOG_CSV}")
        print("  Goodbye! 🧬\n")


# ══════════════════════════════════════════════════════════════════════════════
# ABSTRACT-ONLY FAST PATH (v2 compatible, used when /fulltext is OFF)
# ══════════════════════════════════════════════════════════════════════════════

def search_pubmed_abstractonly(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    """Fast path — abstract only, no PMC lookup. Same as v2 behaviour."""
    try:
        r = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results,
                    "retmode": "json", "sort": "relevance"},
            timeout=10
        )
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        r2 = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids),
                    "rettype": "abstract", "retmode": "xml"},
            timeout=15
        )

        titles    = re.findall(r"<ArticleTitle>(.*?)</ArticleTitle>", r2.text, re.S)
        abstracts = re.findall(r"<AbstractText.*?>(.*?)</AbstractText>", r2.text, re.S)
        years     = re.findall(r"<PubDate>.*?<Year>(\d{4})</Year>", r2.text, re.S)

        return [{
            "pmid":      pmid,
            "title":     titles[i]    if i < len(titles)    else "N/A",
            "abstract":  abstracts[i] if i < len(abstracts) else "No abstract",
            "year":      years[i]     if i < len(years)     else "N/A",
            "doi":       "N/A",
            "pmcid":     None,
            "full_text": None,
            "source":    "ABSTRACT",
        } for i, pmid in enumerate(ids)]

    except Exception as e:
        print(f"  [PubMed error: {e}]")
        return []


if __name__ == "__main__":
    main()
