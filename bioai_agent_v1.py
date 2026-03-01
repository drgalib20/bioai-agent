"""
BioAI Agent v1.2 — Biomedical LLM Comparison with Full-Text Evidence
---------------------------------------------------------------------
Fixes from v1.1:
  - Replaced Mistral-7B-Instruct-v0.3 (no longer a chat model on HF)
    with meta-llama/Llama-3.1-8B-Instruct (confirmed working)
  - PubMed query cleaner: strips output format instructions and typos
    before sending to PubMed API (e.g. "output as docx" removed)
  - Output format detection: agent detects "output as docx/pdf/csv"
    in user prompt and notifies user it cannot generate files directly

Clinical rationale:
  Abstract-only evidence is insufficient for clinical decision making.
  Subgroup effects, safety signals, dosing nuances, and methodological
  limitations are almost always in the full text — not the abstract.
  Full text from PMC is freely available for ~40% of PubMed articles,
  covering most NIH-funded and open-access research.

Evidence pipeline:
  PubMed esearch → PMIDs
  PubMed efetch  → abstracts + metadata
  ID converter   → PMCID lookup per article
  PMC efetch     → full text XML (if available)
  Section parser → Results, Discussion, Conclusions (priority order)
  Fallback       → abstract if no PMC full text

Models (3 general LLMs via HF Inference API — no GPU required):
  1. meta-llama/Llama-3.2-3B-Instruct
  2. Qwen/Qwen2.5-7B-Instruct
  3. meta-llama/Llama-3.1-8B-Instruct

Usage:
  source /mnt/User/python-envs/bioai/bin/activate
  python ~/bioai_agent.py
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

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",    # replaced Mistral (no longer chat model on HF)
]

MAX_TOKENS         = 2048   # full clinical responses
LOG_CSV            = os.path.expanduser("~/bioai_results.csv")
MLFLOW_EXP_NAME    = "bioai_agent"
PUBMED_MAX         = 3
FULLTEXT_MAX_CHARS = 4000   # max chars of full text injected per article
ABSTRACT_MAX_CHARS = 800    # max chars of abstract injected per article

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Output format keywords — stripped from clinical prompt before PubMed search
OUTPUT_FORMAT_PATTERNS = [
    r'\boutput\s+as\s+\w+\b',
    r'\bsave\s+as\s+\w+\b',
    r'\bexport\s+as\s+\w+\b',
    r'\bwrite\s+as\s+\w+\b',
    r'\bformat\s+as\s+\w+\b',
    r'\bin\s+docx\b', r'\bin\s+pdf\b', r'\bin\s+csv\b',
    r'\bas\s+a\s+docx\b', r'\bas\s+a\s+pdf\b',
    r'\bto\s+docx\b', r'\bto\s+pdf\b', r'\bto\s+csv\b',
]

OUTPUT_FORMAT_NAMES = ['docx', 'pdf', 'csv', 'xlsx', 'word', 'spreadsheet']

# ══════════════════════════════════════════════════════════════════════════════
# QUERY PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def detect_output_format(user_input: str) -> str | None:
    """
    Detect if user asked for a specific output format.
    Returns format name string or None.
    e.g. 'output as docx' → 'docx'
    """
    lower = user_input.lower()
    for fmt in OUTPUT_FORMAT_NAMES:
        if fmt in lower:
            return fmt
    return None


def clean_pubmed_query(user_input: str) -> str:
    """
    Strip output format instructions and clean up the query
    before sending to PubMed API.

    Examples:
      'mechanical vs manual CPR output as docx' → 'mechanical vs manual CPR'
      'SGLT2 inhibitors in HF. Save as pdf'     → 'SGLT2 inhibitors in HF'
    """
    cleaned = user_input

    # Remove output format patterns
    for pattern in OUTPUT_FORMAT_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Remove trailing punctuation and extra spaces
    cleaned = re.sub(r'[.!?,;]+$', '', cleaned.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Truncate to 200 chars (PubMed query limit)
    if len(cleaned) > 200:
        cleaned = cleaned[:200]

    return cleaned

# ══════════════════════════════════════════════════════════════════════════════
# PMC FULL TEXT RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def get_pmc_id(pmid: str) -> str | None:
    """
    Convert PubMed PMID → PMC ID using NCBI ID converter.
    Returns 'PMC1234567' or None if article not in PMC.
    """
    try:
        r = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={
                "ids":    pmid,
                "format": "json",
                "tool":   "bioai_agent",
                "email":  "bioai@research.local"
            },
            timeout=10
        )
        records = r.json().get("records", [])
        if records and "pmcid" in records[0]:
            return records[0]["pmcid"]
        return None
    except Exception:
        return None


def fetch_pmc_fulltext(pmcid: str) -> dict:
    """
    Fetch full text XML from PMC and extract named sections.
    Returns dict of {SECTION_TITLE: content_text}.
    """
    try:
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
        sections = {}

        # Extract <sec> blocks with their titles
        sec_blocks = re.findall(
            r'<sec[^>]*>.*?<title>(.*?)</title>(.*?)</sec>',
            xml, re.S | re.I
        )
        for title, content in sec_blocks:
            clean_content = re.sub(r'<[^>]+>', ' ', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            clean_title   = re.sub(r'<[^>]+>', '', title).strip().upper()
            if clean_content and len(clean_content) > 50:
                sections[clean_title] = clean_content

        # Also capture abstract from full text XML
        abstract_match = re.search(
            r'<abstract[^>]*>(.*?)</abstract>', xml, re.S | re.I
        )
        if abstract_match:
            abstract_text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
            sections['ABSTRACT'] = re.sub(r'\s+', ' ', abstract_text).strip()

        return sections

    except Exception:
        return {}


def extract_key_sections(sections: dict, max_chars: int = FULLTEXT_MAX_CHARS) -> str:
    """
    Extract and prioritise clinically relevant sections.
    Priority: Results → Discussion → Conclusions → Abstract → rest.
    Truncates to max_chars total.
    """
    if not sections:
        return ""

    priority = [
        'RESULTS', 'DISCUSSION', 'CONCLUSIONS', 'CONCLUSION',
        'FINDINGS', 'CLINICAL IMPLICATIONS', 'ABSTRACT',
        'INTRODUCTION', 'BACKGROUND', 'METHODS', 'MATERIALS AND METHODS'
    ]

    ordered = []
    seen_contents = set()

    for key in priority:
        for title, content in sections.items():
            if key in title and content not in seen_contents:
                ordered.append((title, content))
                seen_contents.add(content)
                break

    for title, content in sections.items():
        if content not in seen_contents:
            ordered.append((title, content))
            seen_contents.add(content)

    parts = []
    total = 0
    for title, content in ordered:
        if total >= max_chars:
            break
        remaining = max_chars - total
        snippet   = content[:remaining]
        parts.append(f"[{title}]\n{snippet}")
        total    += len(snippet)

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# PUBMED SEARCH — WITH FULL TEXT
# ══════════════════════════════════════════════════════════════════════════════

def search_pubmed_fulltext(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    """
    Search PubMed and attempt full text retrieval from PMC per article.
    Falls back to abstract if full text unavailable.

    Returns list of dicts:
      pmid, title, year, abstract, pmcid, full_text, source
      source = 'FULL TEXT' | 'ABSTRACT'
    """
    try:
        # Step 1: Get PMIDs
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

        # Step 2: Fetch abstracts + metadata
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

        results = []

        for i, pmid in enumerate(ids):
            article = {
                "pmid":      pmid,
                "title":     titles[i]    if i < len(titles)    else "N/A",
                "abstract":  abstracts[i] if i < len(abstracts) else "No abstract available",
                "year":      years[i]     if i < len(years)     else "N/A",
                "pmcid":     None,
                "full_text": None,
                "source":    "ABSTRACT",
            }

            # Step 3: Attempt PMC full text
            print(f"    🔎 PMID {pmid} — checking PMC...", end="", flush=True)
            pmcid = get_pmc_id(pmid)

            if pmcid:
                article["pmcid"] = pmcid
                print(f" {pmcid}, fetching...", end="", flush=True)
                sections = fetch_pmc_fulltext(pmcid)

                if sections:
                    full_text = extract_key_sections(sections)
                    if full_text and len(full_text) > 200:
                        article["full_text"] = full_text
                        article["source"]    = "FULL TEXT"
                        print(f" ✅ {len(full_text)} chars retrieved")
                    else:
                        print(f" ⚠️  Content too short — using abstract")
                else:
                    print(f" ⚠️  Parse failed — using abstract")
            else:
                print(f" ❌ Not in PMC — using abstract")

            results.append(article)

        return results

    except Exception as e:
        print(f"\n  [Evidence retrieval error: {e}]")
        return []


def search_pubmed_abstractonly(query: str, max_results: int = PUBMED_MAX) -> list[dict]:
    """
    Fast path — abstract only, no PMC lookup.
    Used when /fulltext mode is OFF.
    """
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
            "abstract":  abstracts[i] if i < len(abstracts) else "No abstract available",
            "year":      years[i]     if i < len(years)     else "N/A",
            "pmcid":     None,
            "full_text": None,
            "source":    "ABSTRACT",
        } for i, pmid in enumerate(ids)]

    except Exception as e:
        print(f"  [PubMed error: {e}]")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT EVIDENCE CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def format_evidence_context(articles: list[dict]) -> str:
    """
    Format articles into LLM context string.
    Full-text articles get detailed sections;
    abstract-only articles get truncated abstract.
    """
    if not articles:
        return ""

    lines = ["\n=== Clinical Evidence (PubMed/PMC) ==="]

    for a in articles:
        source_tag = f"[{a['source']}]"
        pmcid_tag  = f" | {a['pmcid']}" if a['pmcid'] else ""
        lines.append(f"\n{source_tag} PMID {a['pmid']}{pmcid_tag} ({a['year']})")
        lines.append(f"Title: {a['title']}")

        if a['source'] == "FULL TEXT" and a['full_text']:
            lines.append(f"Full Text (key sections):\n{a['full_text']}")
        else:
            lines.append(f"Abstract: {a['abstract'][:ABSTRACT_MAX_CHARS]}...")

    lines.append("\n=======================================\n")
    return "\n".join(lines)


def evidence_summary(articles: list[dict]) -> str:
    full  = sum(1 for a in articles if a['source'] == 'FULL TEXT')
    abstr = sum(1 for a in articles if a['source'] == 'ABSTRACT')
    parts = []
    if full:
        parts.append(f"📄 {full} full text")
    if abstr:
        parts.append(f"📋 {abstr} abstract only")
    return " | ".join(parts) if parts else "none"


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "turn", "prompt",
                "model", "response", "latency_s",
                "pubmed_pmids", "pmc_ids", "fulltext_count", "abstract_count"
            ])


def log_to_csv(session_id, turn, prompt, model, response, latency, articles):
    pmids     = [a["pmid"]  for a in articles]
    pmc_ids   = [a["pmcid"] for a in articles if a["pmcid"]]
    ft_count  = sum(1 for a in articles if a["source"] == "FULL TEXT")
    abs_count = sum(1 for a in articles if a["source"] == "ABSTRACT")

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            session_id, turn, prompt, model,
            response.replace("\n", " "),
            round(latency, 2),
            "|".join(pmids),
            "|".join(pmc_ids),
            ft_count,
            abs_count,
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
        mlflow.log_param("models",             MODELS)
        mlflow.log_param("pubmed_max",          PUBMED_MAX)
        mlflow.log_param("max_tokens",          MAX_TOKENS)
        mlflow.log_param("fulltext_max_chars",  FULLTEXT_MAX_CHARS)
        print(f"  [MLflow] Run ID: {run.info.run_id}")
        return run
    except Exception as e:
        print(f"  [MLflow unavailable: {e}]")
        return None


def log_to_mlflow(turn, model, latency, response_len, ft_count):
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_metrics({
            f"{model.split('/')[-1]}_latency":      latency,
            f"{model.split('/')[-1]}_response_len": response_len,
            "fulltext_retrieved":                   ft_count,
        }, step=turn)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# QUERY A SINGLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def query_model(client: InferenceClient, model: str,
                history: list[dict], prompt: str,
                evidence_context: str) -> tuple[str, float]:
    """Send prompt + history + evidence to model, return (response, latency)."""

    messages     = list(history)
    user_content = prompt

    if evidence_context:
        user_content = (
            evidence_context +
            "\nUsing the clinical evidence above (full text where available), "
            "answer this question thoroughly:\n" +
            prompt
        )
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

W       = 100  # wider display — reduces visual truncation of long lines
DIVIDER = "═" * W
DIV2    = "─" * W


def print_banner():
    print(f"\n{DIVIDER}")
    print("  🧬  BioAI Agent v1.2 — Full-Text Evidence + LLM Comparison")
    print(f"{DIVIDER}")
    print("  Evidence: PMC full text where available, abstract fallback")
    print("  Commands:")
    print("    /models    — list active models")
    print("    /pubmed    — toggle evidence retrieval on/off")
    print("    /fulltext  — toggle full-text vs abstract-only mode")
    print("    /evidence  — show last retrieved evidence sources")
    print("    /history   — show conversation history")
    print("    /clear     — clear conversation history")
    print("    /save      — show path to results CSV")
    print("    /quit      — exit")
    print(f"{DIVIDER}\n")


def print_response(model: str, text: str, latency: float):
    short = model.split("/")[-1]
    print(f"\n  ┌─ {short} ({latency:.1f}s) {'─'*(W - len(short) - 10)}")
    for line in text.split("\n"):
        if line.strip():
            while len(line) > W - 6:
                print(f"  │  {line[:W-6]}")
                line = "    " + line[W-6:]
            print(f"  │  {line}")
    print(f"  └{'─'*(W-2)}")


def print_evidence_list(articles: list[dict]):
    if not articles:
        print("  No evidence retrieved yet.")
        return
    print(f"\n  📚 Last retrieved evidence ({len(articles)} articles):")
    for a in articles:
        icon  = "📄" if a['source'] == 'FULL TEXT' else "📋"
        pmcid = f" | {a['pmcid']}" if a['pmcid'] else ""
        print(f"  {icon} [{a['year']}] PMID {a['pmid']}{pmcid}")
        print(f"     {a['source']}: {a['title'][:65]}")
    print()


def print_session_stats(stats: dict):
    total = stats["total_articles"]
    if total == 0:
        return
    pct = (stats["fulltext"] / total) * 100
    print(f"\n  📈 Session Evidence Stats:")
    print(f"     Total articles:  {total}")
    print(f"     Full text (PMC): {stats['fulltext']}  ({pct:.0f}%)")
    print(f"     Abstract only:   {stats['abstract_only']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AGENT LOOP
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
    use_fulltext  = True          # full text ON by default
    active_models = list(MODELS)
    last_articles = []            # for /evidence command
    session_stats = {             # session-level evidence tracking
        "total_articles": 0,
        "fulltext":        0,
        "abstract_only":   0
    }

    init_csv()
    mlflow_run = init_mlflow()
    print_banner()

    print(f"  Active models ({len(active_models)}):")
    for m in active_models:
        print(f"    • {m}")
    print(f"\n  Evidence retrieval:   {'✅ ON' if use_pubmed else '❌ OFF'}")
    print(f"  Full-text mode (PMC): {'✅ ON' if use_fulltext else '❌ OFF'}")
    print(f"  Results logged to:    {LOG_CSV}")
    print(f"\n  ⚠️  Full-text adds ~3-8s per article. Use /fulltext to toggle.\n")

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
                print("\n  Active models:")
                for m in active_models:
                    print(f"    • {m}")
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

            elif cmd == "/evidence":
                print_evidence_list(last_articles)
                continue

            elif cmd == "/history":
                if not history:
                    print("  No history yet.")
                for msg in history:
                    print(f"  [{msg['role'].upper()}] {msg['content'][:120]}...")
                continue

            elif cmd == "/clear":
                history.clear()
                turn = 0
                print("  ✅ History cleared.")
                continue

            elif cmd == "/save":
                print(f"  CSV saved at: {LOG_CSV}")
                continue

            # ── Output format detection ───────────────────────────────
            output_fmt = detect_output_format(user_input)
            if output_fmt:
                print(f"\n  ℹ️  Output format '{output_fmt}' detected.")
                print(f"  The agent cannot generate files directly.")
                print(f"  Responses will be printed to terminal.")
                print(f"  💡 Tip: Copy the response text into your preferred")
                print(f"          editor, or use the CSV log at {LOG_CSV}")

            # Clean clinical query for PubMed (strip output instructions)
            pubmed_query = clean_pubmed_query(user_input)
            if pubmed_query != user_input.strip():
                print(f"\n  🔧 PubMed query cleaned: '{pubmed_query[:70]}'")

            # ── Evidence Retrieval ────────────────────────────────────────
            articles         = []
            evidence_context = ""

            if use_pubmed:
                print(f"\n  🔍 Searching: '{pubmed_query[:60]}'...")

                if use_fulltext:
                    print(f"  📄 Full-text mode — checking PMC for each article...")
                    articles = search_pubmed_fulltext(pubmed_query)
                else:
                    articles = search_pubmed_abstractonly(pubmed_query)

                if articles:
                    last_articles    = articles
                    evidence_context = format_evidence_context(articles)

                    # Update session stats
                    session_stats["total_articles"] += len(articles)
                    for a in articles:
                        if a["source"] == "FULL TEXT":
                            session_stats["fulltext"]      += 1
                        else:
                            session_stats["abstract_only"] += 1

                    print(f"\n  📚 Evidence: {evidence_summary(articles)}")
                    for a in articles:
                        icon  = "📄" if a['source'] == 'FULL TEXT' else "📋"
                        pmcid = f" ({a['pmcid']})" if a['pmcid'] else ""
                        print(f"    {icon} [{a['year']}] PMID {a['pmid']}{pmcid}: {a['title'][:55]}")
                else:
                    print("  ⚠️  No evidence found.")

            ft_count = sum(1 for a in articles if a["source"] == "FULL TEXT")
            turn += 1
            print(f"\n{DIV2}")
            print(f"  Turn {turn} — {len(active_models)} model(s) | "
                  f"Evidence {'ON' if use_pubmed else 'OFF'} | "
                  f"{'Full text' if use_fulltext else 'Abstract only'}")
            print(DIV2)

            # ── Query each model ──────────────────────────────────────────
            responses = {}
            for model in active_models:
                short = model.split("/")[-1]
                print(f"  ⏳ {short}...", end="", flush=True)
                text, latency = query_model(
                    client, model, history, user_input, evidence_context
                )
                responses[model] = (text, latency)
                print(f" ✅ {latency:.1f}s")

                log_to_csv(session_id, turn, user_input,
                           model, text, latency, articles)
                log_to_mlflow(turn, model, latency, len(text), ft_count)

            # ── Print all responses ───────────────────────────────────────
            for model, (text, latency) in responses.items():
                print_response(model, text, latency)

            # ── Update history ────────────────────────────────────────────
            history.append({"role": "user",      "content": user_input})
            first_response = list(responses.values())[0][0]
            history.append({"role": "assistant", "content": first_response})

            print(f"\n  📄 Logged to: {LOG_CSV}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()
        print_session_stats(session_stats)
        print(f"\n  Session ended. Results saved to: {LOG_CSV}")
        print("  Goodbye! 🧬\n")


if __name__ == "__main__":
    main()
