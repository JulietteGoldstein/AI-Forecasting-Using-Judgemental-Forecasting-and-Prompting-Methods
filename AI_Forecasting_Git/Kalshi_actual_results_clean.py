"""
Kalshi Actual Results Cleaner

This module cleans and repairs actual market results data by:
1. Fetching missing data from Kalshi API endpoints
2. Deriving actual answers from market resolution data
3. Removing duplicates (keeping best resolution quality)
4. Dropping rows without resolved actual answers (optional)

The module handles binary markets, multi-outcome markets, and ladder-style markets
with special inference logic for each type.

Configuration:
- INPUTS: List of (input_csv, output_csv, dropped_log_csv) tuples to process
- DROP_IF_STILL_MISSING_ACTUAL_ANSWER: Whether to exclude unresolved events
"""

import pandas as pd
import requests
import time
import re
from pathlib import Path

# ========= CONFIG =========
INPUTS = [
    ("your_actual_results_dataset.csv", "cleaned_actual_results.csv", "dropped_actual_results.csv"),
]

BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_TIMEOUT_SECONDS = 20
SLEEP_BETWEEN_CALLS_SECONDS = 0.05

EVENT_TICKER_COL = "event_ticker"

COLS = [
    "event_ticker",
    "prompt",
    "answer_options",
    "actual_answer",
    "actual_market_ticker",
    "market_result",
    "settlement_ts",
    "market_status",
    "debug_note",
]

# If you want to keep "unknown" rows instead of dropping them, set this to False.
DROP_IF_STILL_MISSING_ACTUAL_ANSWER = True

# ===== CUSTOMIZATION NOTES =====
# To clean different datasets:
# 1. Add tuples to INPUTS list with your (input_csv, output_csv, dropped_csv) paths
# 2. Adjust SLEEP_BETWEEN_CALLS_SECONDS if API rate limits (increase) or too slow (decrease)
# 3. Set DROP_IF_STILL_MISSING_ACTUAL_ANSWER = False to keep unresolved rows
# ==================================

# ========= HELPERS =========

def safe_get(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def fetch_market_detail(market_ticker: str) -> dict | None:
    payload = safe_get(f"{BASE}/markets/{market_ticker}")
    if not payload:
        return None
    if isinstance(payload.get("market"), dict):
        return payload["market"]
    return payload

def fetch_event(event_ticker: str) -> dict | None:
    return safe_get(f"{BASE}/events/{event_ticker}")

def fetch_markets_for_event(event_ticker: str, limit: int = 1000) -> list[dict]:
    payload = safe_get(f"{BASE}/markets", params={"event_ticker": event_ticker, "limit": limit})
    if not payload:
        return []
    mkts = payload.get("markets") or []
    if not isinstance(mkts, list):
        return []
    # enforce local filter (sometimes servers ignore params)
    mkts = [m for m in mkts if str(m.get("event_ticker", "")).strip() == str(event_ticker).strip()]
    return mkts

def normalize_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def split_answer_options(options_str: str) -> list[str]:
    return [o.strip() for o in str(options_str or "").split(",") if o.strip()]

def normalize_market_label(m: dict) -> str:
    # best label choice for multi-market winner
    for k in ("subtitle", "yes_sub_title", "title"):
        v = (m.get(k) or "").strip()
        if v:
            return v
    return (m.get("ticker") or "UNKNOWN").strip()

def binary_yes_no_labels(m: dict) -> tuple[str, str]:
    yes_raw = (m.get("yes_sub_title") or "").strip() or (m.get("subtitle") or "").strip() or (m.get("title") or "").strip() or "Yes"
    no_raw = (m.get("no_sub_title") or "").strip()

    if not no_raw or no_raw.lower() == yes_raw.lower():
        # fallback "Not X"
        if yes_raw.lower().startswith("not "):
            no_label = yes_raw[4:].strip()
        else:
            no_label = f"Not {yes_raw}"
    else:
        no_label = no_raw

    return yes_raw, no_label

def infer_bundle_all_no(answer_options: str) -> str:
    """
    For ladder-style markets where ALL binary thresholds resolved NO.
    In many Kalshi “Above X” ladders, that implies "Not Above lowest threshold".
    We pick the *lowest* threshold-looking option and negate it.
    """
    opts = split_answer_options(answer_options)
    if not opts:
        return ""

    # Prefer an option like "Above 1 inch" / "Over 3.5 goals" etc.
    # Pick "lowest" by extracting first number; fallback to first option.
    def extract_num(s: str):
        m = re.search(r"(-?\d+(\.\d+)?)", s)
        return float(m.group(1)) if m else None

    numbered = []
    for o in opts:
        n = extract_num(o)
        if n is not None:
            numbered.append((n, o))
    if numbered:
        numbered.sort(key=lambda x: x[0])
        lowest_opt = numbered[0][1]
    else:
        lowest_opt = opts[0]

    # try a clean "Not <option>" that matches your dataset style
    # If the dataset uses "Not Above 1 inch", this will match.
    if lowest_opt.lower().startswith(("above ", "over ")):
        return f"Not {lowest_opt}"
    return f"Not {lowest_opt}"

def derive_actual_answer_from_market(market: dict, answer_options: str) -> tuple[str, str]:
    """
    Returns (actual_answer, market_result_label)
    market_result_label is left as-is; we mainly care about actual_answer.
    """
    if not market:
        return ("", "")

    mtype = (market.get("market_type") or "").strip().lower()
    res = (market.get("result") or "").strip().lower()
    status = (market.get("status") or "").strip()

    # Binary
    if mtype == "binary":
        yes_label, no_label = binary_yes_no_labels(market)

        if res == "yes":
            return (yes_label, "yes")
        if res == "no":
            return (no_label, "no")

        # Special-case: sometimes you stored a synthetic market_result like bundle_all_no_inferred
        # But market detail itself might still be yes/no/void/invalid.
        if res in ("void", "invalid"):
            return (res, res)

        return ("", res)

    # Multi outcome market: winner has result == "yes"
    # If this is a single market in a bundle, it might still show yes/no; then label is fine.
    if res == "yes":
        return (normalize_market_label(market), "yes")

    # otherwise unknown here
    return ("", res)

def derive_actual_answer_from_event(event_ticker: str, answer_options: str) -> tuple[str, str, str]:
    """
    Try to derive actual_answer using the event->markets path.
    Returns (actual_answer, actual_market_ticker, market_result)
    """
    mkts = fetch_markets_for_event(event_ticker)
    time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

    if not mkts:
        return ("", "", "")

    # Fetch details because list endpoint can omit result/status
    detailed = []
    for m in mkts:
        t = (m.get("ticker") or "").strip()
        if not t:
            continue
        d = fetch_market_detail(t)
        time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)
        if d:
            detailed.append(d)

    if not detailed:
        return ("", "", "")

    winners = [m for m in detailed if str(m.get("result") or "").strip().lower() == "yes"]
    if winners:
        w = winners[0]
        return (normalize_market_label(w), str(w.get("ticker") or "").strip(), "yes")

    # If no yes winners, but every market is a binary threshold and all are "no",
    # infer "Not <lowest threshold>" style.
    results = [str(m.get("result") or "").strip().lower() for m in detailed if str(m.get("result") or "").strip()]
    if results and all(r == "no" for r in results):
        inferred = infer_bundle_all_no(answer_options)
        # choose a representative market ticker (lowest strike often matches suffix)
        rep = str(detailed[0].get("ticker") or "").strip()
        return (inferred, rep, "bundle_all_no_inferred")

    return ("", "", "")

def resolution_quality_score(row: pd.Series) -> int:
    score = 0
    aa = str(row.get("actual_answer", "")).strip()
    mt = str(row.get("actual_market_ticker", "")).strip()
    mr = str(row.get("market_result", "")).strip()
    ms = str(row.get("market_status", "")).strip().lower()
    st = str(row.get("settlement_ts", "")).strip()
    dbg = str(row.get("debug_note", "")).strip()

    if aa:
        score += 100
    if mt:
        score += 10
    if mr:
        score += 5
    if st:
        score += 2
    if ms in ("finalized", "settled"):
        score += 2
    if aa and not dbg:
        score += 1
    return score

def dedupe_keep_best(df: pd.DataFrame) -> pd.DataFrame:
    if EVENT_TICKER_COL not in df.columns:
        return df
    df = df.copy()
    df["_score"] = df.apply(resolution_quality_score, axis=1)
    df = df.sort_values(by=[EVENT_TICKER_COL, "_score"], ascending=[True, False])
    df = df.drop_duplicates(subset=[EVENT_TICKER_COL], keep="first").drop(columns=["_score"])
    return df.reset_index(drop=True)

# ========= MAIN CLEAN/REPAIR =========

def repair_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for i, row in df.iterrows():
        aa = str(row.get("actual_answer", "")).strip()
        if aa:
            continue

        event_ticker = str(row.get("event_ticker", "")).strip()
        market_ticker = str(row.get("actual_market_ticker", "")).strip()
        answer_options = str(row.get("answer_options", "")).strip()
        market_result_existing = str(row.get("market_result", "")).strip()

        # 1) Best: derive from recorded actual_market_ticker
        if market_ticker:
            m = fetch_market_detail(market_ticker)
            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)
            if m:
                derived_aa, derived_mr = derive_actual_answer_from_market(m, answer_options)
                if derived_aa:
                    df.at[i, "actual_answer"] = derived_aa
                    # keep your existing market_result unless it’s blank
                    if not market_result_existing and derived_mr:
                        df.at[i, "market_result"] = derived_mr
                    # repair missing settlement/status too
                    if not str(row.get("settlement_ts", "")).strip():
                        df.at[i, "settlement_ts"] = str(m.get("settlement_ts") or "").strip()
                    if not str(row.get("market_status", "")).strip():
                        df.at[i, "market_status"] = str(m.get("status") or "").strip()
                    continue

        # 2) Fallback: derive from the event’s markets
        if event_ticker:
            derived_aa, derived_mt, derived_mr = derive_actual_answer_from_event(event_ticker, answer_options)
            if derived_aa:
                df.at[i, "actual_answer"] = derived_aa
                if not market_ticker and derived_mt:
                    df.at[i, "actual_market_ticker"] = derived_mt
                if not market_result_existing and derived_mr:
                    df.at[i, "market_result"] = derived_mr
                continue

        # 3) If your pipeline already labeled bundle_all_no_inferred, infer from answer_options without API
        if market_result_existing == "bundle_all_no_inferred":
            inferred = infer_bundle_all_no(answer_options)
            if inferred:
                df.at[i, "actual_answer"] = inferred
                continue

    return df

def clean_and_write(in_path: str, out_path: str, dropped_path: str) -> None:
    if not Path(in_path).exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)

    for c in COLS:
        if c not in df.columns:
            df[c] = ""

    df = df[COLS].copy()
    df = normalize_str_cols(df)

    before = len(df)

    # Repair missing actual_answer
    df = repair_rows(df)

    # Dedupe
    df = dedupe_keep_best(df)
    after_dedup = len(df)

    # Drop anything that STILL has no actual_answer (per your requirement)
    if DROP_IF_STILL_MISSING_ACTUAL_ANSWER:
        missing_mask = df["actual_answer"].fillna("").astype(str).str.strip() == ""
        dropped = df[missing_mask].copy()
        kept = df[~missing_mask].copy()

        dropped["drop_reason"] = "missing_actual_answer_after_repair"
        kept.to_csv(out_path, index=False)
        dropped.to_csv(dropped_path, index=False)
    else:
        kept = df
        dropped = df.iloc[0:0].copy()
        kept.to_csv(out_path, index=False)
        dropped.to_csv(dropped_path, index=False)

    print(f"\n=== {in_path} ===")
    print(f"Rows in: {before}")
    print(f"After dedupe: {after_dedup}")
    print(f"Kept (actual_answer present): {len(kept)}")
    print(f"Dropped (still missing actual_answer): {len(dropped)}")
    print(f"Wrote: {out_path}")
    print(f"Wrote dropped log: {dropped_path}")

def main():
    for in_path, out_path, dropped_path in INPUTS:
        clean_and_write(in_path, out_path, dropped_path)

if __name__ == "__main__":
    main()
