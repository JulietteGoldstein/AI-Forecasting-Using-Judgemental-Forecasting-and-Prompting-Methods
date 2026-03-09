"""
Kalshi Event Data Collector

This script fetches prediction market events from the Kalshi API, augments them with
interpretations via OpenAI, and exports the data for forecasting model evaluation.

The script collects events expiring on specified target dates, normalizes market
labels and answer options, and optionally interprets questions to clarify the
underlying forecasting task.

SETUP REQUIREMENTS:
  - Kalshi API credentials (ACCESS_KEY and private key file)
  - OpenAI API key in .env as OPENAI_API_KEY2
  - Private key at ./.venv/kalshi-key.key

CUSTOMIZATION:
  - Edit TARGET_DATES to specify when events should expire
  - Adjust NUM_QUESTIONS for batch size
  - Set STATUS_FILTER to "open", "closed", "settled", or None
  - Modify OUTPUT_CSV and OUTPUT_DEBUG to change output paths
  - Remove AI interpretation if not needed (set openai_client = None)
"""

import time
import requests
import pandas as pd
from datetime import datetime
from collections import defaultdict
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import os
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# KALSHI API AUTHENTICATION
# ============================================================================

KALSHI_ACCESS_KEY = "d27a6d05-f658-45d3-87e0-df27af9ed63d"
PRIVATE_KEY_PATH = './.venv/kalshi-key.key'

with open(PRIVATE_KEY_PATH, "rb") as f:
    PRIVATE_KEY = serialization.load_pem_private_key(f.read(), password=None)


def build_auth_headers(method: str, path: str) -> dict:
    """
    Generate signed authentication headers for Kalshi API requests.
    
    Signs message with: timestamp + HTTP_METHOD + /trade-api/v2{path}
    Required for accessing restricted endpoints like forecast percentile history.
    
    Args:
        method: HTTP method (e.g., "GET", "POST")
        path: API path after /trade-api/v2 (e.g., "/series/ABC/events/DEF/forecast_percentile_history")
    
    Returns:
        Dict with KALSHI-ACCESS-* headers for request authentication
    """
    ts_ms = str(int(time.time() * 1000))
    path_to_sign = "/trade-api/v2" + path
    message = f"{ts_ms}{method}{path_to_sign}".encode("utf-8")

    signature_bytes = PRIVATE_KEY.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    signature = base64.b64encode(signature_bytes).decode("utf-8")

    return {
        "KALSHI-ACCESS-KEY": KALSHI_ACCESS_KEY,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Edit these values for your dataset
NUM_QUESTIONS = 700
TARGET_DATES = {"2025-12-31", "2026-01-01"}  # Specify expiration dates to collect
OUTPUT_CSV = "your_events_dataset.csv"
OUTPUT_DEBUG = "your_events_dataset_debug.json"

PAGE_LIMIT = 1000
STATUS_FILTER = "open"  # Set to None to fetch all statuses

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_markets_page(cursor: str | None = None) -> dict:
    """
    Fetch a paginated batch of markets from Kalshi API.
    
    Args:
        cursor: Pagination cursor from previous request (None for first page)
    
    Returns:
        API response dict containing markets list and next cursor
    """
    params = {"limit": PAGE_LIMIT}
    if cursor:
        params["cursor"] = cursor
    if STATUS_FILTER:
        params["status"] = STATUS_FILTER
    r = requests.get(f"{BASE}/markets", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def iso_to_date_str(iso: str | None) -> str | None:
    """Extract YYYY-MM-DD date from ISO 8601 timestamp."""
    if not iso:
        return None
    return iso[:10]


def parse_float(x) -> float | None:
    """Safely convert value to float, returning None on failure."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def normalize_market_label(m: dict) -> str:
    """
    Extract a human-readable label for a market outcome.
    
    For multi-market events, each market represents an outcome (candidate, price range, etc.).
    Prioritizes: subtitle > yes_sub_title > title > ticker
    
    Args:
        m: Market dict from Kalshi API
    
    Returns:
        Human-readable market label
    """
    for k in ("subtitle", "yes_sub_title", "title"):
        v = (m.get(k) or "").strip()
        if v:
            return v
    return m.get("ticker", "UNKNOWN")


def roll_up_event(markets: list[dict]) -> dict:
    """
    Aggregate multiple markets into a single event representation.
    
    For binary events: creates yes/no answer options
    For multi-outcome events: labels each market outcome
    
    Also extracts timing, popularity metrics (most popular option and open interest),
    and per-market snapshots for downstream analysis.
    
    Args:
        markets: List of market dicts from Kalshi API for a single event
    
    Returns:
        Dict with normalized event structure: series_ticker, event_ticker, answer_options, etc.
    """
    # Extract time bounds
    open_times = [m.get("open_time") for m in markets if m.get("open_time")]
    exp_times = [
        m.get("expiration_time") or m.get("latest_expiration_time")
        for m in markets
        if (m.get("expiration_time") or m.get("latest_expiration_time"))
    ]
    start_date = min(open_times) if open_times else None
    end_date = max(exp_times) if exp_times else None

    # ===== Build answer options =====
    if len(markets) == 1 and (markets[0].get("market_type") == "binary"):
        # Binary event: extract yes/no labels
        m = markets[0]
        yes_raw = (m.get("yes_sub_title") or "").strip()
        if not yes_raw:
            yes_raw = (m.get("subtitle") or "").strip()
        if not yes_raw:
            yes_raw = (m.get("title") or "").strip()
        if not yes_raw:
            yes_raw = "Yes"

        no_raw = (m.get("no_sub_title") or "").strip()
        yes_label = yes_raw

        # Generate NO label if missing or identical to YES
        if not no_raw or no_raw.lower() == yes_label.lower():
            if yes_label.lower().startswith("not "):
                no_label = yes_label[4:].strip()
            else:
                no_label = f"Not {yes_label}"
        else:
            no_label = no_raw

        answer_options = [yes_label, no_label]

    else:
        # Multi-market event: each market is one outcome (candidate, price range, etc.)
        seen = set()
        answer_options = []
        for m in markets:
            lbl = normalize_market_label(m)
            if lbl and lbl not in seen:
                answer_options.append(lbl)
                seen.add(lbl)

    # ===== Market popularity (human proxy) =====
    # For binary events, all open interest is in a single market
    # For multi-outcome events, identify which outcome has most interest
    def get_oi(m): return m.get("open_interest") or 0
    
    top_mkt = max(markets, key=get_oi, default=None)
    if top_mkt is not None:
        if len(markets) == 1 and markets[0].get("market_type") == "binary":
            most_popular_option = answer_options[0] if answer_options else None
        else:
            most_popular_option = normalize_market_label(top_mkt)
        popularity_metric = get_oi(top_mkt)

    else:
        most_popular_option = None
        popularity_metric = 0

    # Per-market snapshot for debugging/analysis
    per_option = []
    for m in markets:
        yes_bid = parse_float(m.get("yes_bid_dollars"))
        yes_ask = parse_float(m.get("yes_ask_dollars"))
        mid_yes = None
        if yes_bid is not None and yes_ask is not None:
            mid_yes = (yes_bid + yes_ask) / 2

        per_option.append({
            "market_label": normalize_market_label(m),
            "yes_sub_title": m.get("yes_sub_title"),
            "no_sub_title": m.get("no_sub_title"),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_mid": mid_yes,
            "open_interest": m.get("open_interest"),
            "volume_24h": m.get("volume_24h"),
            "ticker": m.get("ticker"),
            "status": m.get("status"),
        })

    first = markets[0]
    return {
        "series_ticker": first.get("series_ticker"),
        "event_ticker": first.get("event_ticker"),
        "category": first.get("category"),
        "start_date": start_date,
        "end_date": end_date,
        "num_markets": len(markets),
        "answer_options": answer_options,
        "most_popular_option": most_popular_option,
        "most_popular_open_interest": popularity_metric,
        "total_open_interest": sum((m.get("open_interest") or 0) for m in markets),
        "total_volume_24h": sum((m.get("volume_24h") or 0) for m in markets),
        "markets_snapshot": per_option,
    }


def fetch_event_title(series_ticker: str | None, event_ticker: str | None) -> str | None:
    """
    Fetch canonical event question from the Kalshi /series/{id}/events/{id} endpoint.
    
    This is more reliable than extracting from market titles.
    
    Args:
        series_ticker: Series identifier (e.g., "ELECTION2024")
        event_ticker: Event identifier (e.g., "WINNER")
    
    Returns:
        Event title/question, or None if unavailable
    """
    if not series_ticker or not event_ticker:
        return None

    path = f"/series/{series_ticker}/events/{event_ticker}"
    url = BASE + path
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        return (
            (data.get("title") or "").strip()
            or (data.get("event_title") or "").strip()
            or (data.get("short_name") or "").strip()
            or None
        )
    except Exception as e:
        print(f"Warning: failed to fetch event title for {series_ticker}/{event_ticker}: {e!r}")
        return None


def get_event_p50_forecast(series_ticker: str, event_ticker: str) -> float | None:
    """
    Fetch market forecast median (p50) for an event from the last 30 days.
    
    Useful for capturing market consensus over time.
    Requires authentication headers.
    
    Args:
        series_ticker: Series identifier
        event_ticker: Event identifier
    
    Returns:
        Median forecast value, or None if unavailable
    """
    now_ms = int(time.time() * 1000)
    day_ms = 24 * 60 * 60 * 1000
    start_ts = now_ms - 30 * day_ms
    end_ts = now_ms

    params = {
        "percentiles": [5000],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1440,
    }

    path = f"/series/{series_ticker}/events/{event_ticker}/forecast_percentile_history"
    headers = build_auth_headers("GET", path)

    r = requests.get(BASE + path, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    history = payload.get("forecast_history", [])
    if not history:
        return None

    last_point = history[-1]
    points = last_point.get("percentile_points", [])
    for p in points:
        if p.get("percentile") == 5000:
            return p.get("raw_numerical_forecast") or p.get("numerical_forecast")

    return None


# ============================================================================
# QUESTION INTERPRETATION (optional AI enhancement)
# ============================================================================

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY2")
if not OPENAI_KEY:
    print("WARNING: No OPENAI_API_KEY2 found. Questions will not be interpreted.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_KEY)

INTERPRET_SYSTEM_PROMPT = """
You rewrite prediction market questions into clear, underlying questions in plain English.

If the question appears to be specific to one of the answer options, rewrite the QUESTION into the true underlying question.

Example:
- QUESTION: "Will Trump meet with Kim Jong-Un before 2025?"
  ANSWER_OPTIONS: "Kim Jong-Un, Justin Trudeau"
  TRUE UNDERLYING QUESTION: "Who will Trump meet with before 2025?"

If the question is already a general, well-formed question, keep it as is.

Important:
- Return ONLY the rewritten question text.
- Do NOT include labels, explanations, or anything else.
"""


def interpret_question_with_ai(question: str, answer_options: str) -> str:
    """
    Use OpenAI to clarify question into its true underlying form.
    
    AI forecasting often works better when the question is framed as a
    genuinely open prediction task rather than being specific to one answer option.
    
    Example: Market may ask "Will Trump meet with Kim Jong-Un?"
    with options ["Kim Jong-Un", "Justin Trudeau"], but the true question
    is "Who will Trump meet with before 2025?"
    
    CUSTOMIZATION NOTE: Remove this function or set openai_client = None
    if you don't want AI question interpretation.
    
    Args:
        question: Market question as stated
        answer_options: Comma-separated list of options
    
    Returns:
        Clarified question (or original if AI call fails)
    """
    if not openai_client:
        return question

    q = (question or "").strip()
    opts = (answer_options or "").strip()

    if not q:
        return q

    user_msg = f"""
QUESTION: {q}
ANSWER_OPTIONS: {opts}

First, IF the question appears to be specific to one of the answer options, rewrite the QUESTION into the true underlying question in plain English.
(Example: If QUESTION = "Will Trump meet with Kim Jong-Un before 2025?"
 and ANSWER_OPTIONS = "Kim Jong-Un, Justin Trudeau",
 then the true question is "Who will Trump meet with before 2025?")

If it is already a clean, general question, keep it as is.

Return ONLY the rewritten question text. No labels, no explanation, no quotes.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": INTERPRET_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: interpretation failed for question '{q[:50]}...': {e}")
        return question


# ============================================================================
# MAIN COLLECTION PIPELINE
# ============================================================================

def collect_events(num_questions: int, target_dates: set[str], output_csv: str, output_debug: str):
    """
    Main pipeline: fetch markets, aggregate into events, and save to CSV.
    
    Steps:
      1. Paginate through all markets, filtering by target expiration dates
      2. Group markets by event_ticker
      3. Aggregate each group into a unified event representation
      4. Fetch canonical event titles and forecast metadata
      5. Optionally interpret questions via AI
      6. Export to CSV (main) and JSON (detailed debugging)
    
    Args:
        num_questions: Target number of events to collect
        target_dates: Set of expiration dates (YYYY-MM-DD) to filter for
        output_csv: Path to save main events CSV
        output_debug: Path to save detailed JSON snapshot
    """
    cursor = None
    markets_by_event = defaultdict(list)

    print(f"Fetching markets expiring on {target_dates}...")
    while True:
        page = get_markets_page(cursor)
        for m in page.get("markets", []):
            exp_date = iso_to_date_str(
                m.get("expiration_time") or m.get("latest_expiration_time")
            )
            if exp_date in target_dates:
                markets_by_event[m["event_ticker"]].append(m)

        if len(markets_by_event) >= num_questions:
            break

        cursor = page.get("cursor")
        if not cursor:
            break

    print(f"Found {len(markets_by_event)} distinct events")

    rows = []
    for ev_ticker, mkts in markets_by_event.items():
        ev_info = roll_up_event(mkts)
        series_ticker = mkts[0].get("series_ticker")
        question = None

        # Try primary source: dedicated events endpoint
        event_title = fetch_event_title(series_ticker, ev_ticker)
        if event_title:
            question = event_title
        else:
            # Fallbacks using market data
            explicit_event_title = mkts[0].get("event_title")
            if explicit_event_title:
                question = explicit_event_title
            else:
                titles = {
                    (m.get("title") or "").strip()
                    for m in mkts
                    if (m.get("title") or "").strip()
                }
                if len(titles) == 1:
                    question = next(iter(titles))
                else:
                    question = (mkts[0].get("title") or "").strip() or None

        ev_info["question"] = question
        rows.append(ev_info)

    # Sort by expiration and limit to target count
    rows.sort(key=lambda r: (r["end_date"] or ""))
    rows = rows[:num_questions]

    # Fetch forecast percentiles
    for r in rows:
        series_ticker = r.get("series_ticker")
        event_ticker = r.get("event_ticker")
        if series_ticker and event_ticker:
            try:
                r["p50_forecast"] = get_event_p50_forecast(series_ticker, event_ticker)
            except Exception as e:
                print(f"Warning: forecast fetch failed for {series_ticker}/{event_ticker}: {e!r}")
                r["p50_forecast"] = None
        else:
            r["p50_forecast"] = None

    # Flatten for CSV export
    flat = []
    for r in rows:
        flat.append({
            "series_ticker": r["series_ticker"],
            "event_ticker": r["event_ticker"],
            "category": r["category"],
            "question": r["question"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "num_markets": r["num_markets"],
            "answer_options": ", ".join(r["answer_options"]) if r["answer_options"] else None,
            "most_popular_option": r.get("most_popular_option"),
            "most_popular_open_interest": r["most_popular_open_interest"],
            "total_open_interest": r["total_open_interest"],
            "total_volume_24h": r["total_volume_24h"],
            "p50_forecast": r.get("p50_forecast"),
        })

    df = pd.DataFrame(flat)

    # Add AI-interpreted questions
    interpreted_list = []
    for idx, row in df.iterrows():
        q = row.get("question", "")
        opts = row.get("answer_options", "") or ""
        interpreted_q = interpret_question_with_ai(q, opts)
        interpreted_list.append(interpreted_q)

    df["interpreted_question"] = interpreted_list

    # Display and save
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    if df.empty:
        print(f"No events matched expiration dates: {target_dates}")
    else:
        print("\nPreview of collected events:")
        print(df.to_string(index=False))

    # Save outputs
    df.to_csv(output_csv, index=False)
    pd.DataFrame(rows).to_json(output_debug, orient="records", indent=2)

    print(f"\n✓ Saved {len(df)} events to: {output_csv}")
    print(f"✓ Saved detailed snapshot to: {output_debug}")


def main():
    """Entry point: collect events with specified configuration."""
    collect_events(NUM_QUESTIONS, TARGET_DATES, OUTPUT_CSV, OUTPUT_DEBUG)


if __name__ == "__main__":
    main()
