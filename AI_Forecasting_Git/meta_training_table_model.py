"""
Meta Training Table Model

Implements meta-learning for forecasting by training on historical model performances.
Creates a training table from past predictions and actual outcomes, then uses this
to inform future predictions by weighting base models according to their empirical accuracy.

Key methodology:
- Builds training table from historical events with model predictions and actual outcomes
- Computes smoothed accuracy scores for each base model (scenario, prompt engineering, etc.)
- Uses meta-reasoning to combine base model predictions weighted by their track record
- Applies smoothing to handle limited training data

Configuration:
- TRAIN_EVENTS_CSV: Historical events for training
- ACTUAL_RESULTS_CSV: Actual outcomes for training events
- PREDICT_EVENTS_CSV: New events to predict on
- Various model output CSVs: Predictions from base models
- OUTPUT_PREDICTIONS_CSV: Final meta-model predictions
"""

import os
import re
import json
import time
import random
from typing import Dict, Optional, Tuple, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ==============================
# CONFIG — EDIT THESE FILENAMES
# ==============================

TRAIN_EVENTS_CSV = "your_training_events_dataset.csv"
ACTUAL_RESULTS_CSV = "your_actual_results_dataset.csv"

PREDICT_EVENTS_CSV = "your_prediction_events_dataset.csv"

SCENARIO_CSV = "scenario_forecasting_model_outputs.csv"
PROMPT_ENG_CSV = "prompt_engineering_forecasting_outputs.csv"
MOST_POP_CSV = "most_popular_option_model_outputs.csv"
CONTROL_CSV = "control_model_outputs.csv"
DELPHI_CSV = "delphi_outputs.csv"
ENSEMBLE5_CSV = "ensemble5_model_outputs.csv"

OUT_TRAINING_TABLE_CSV = "meta_training_table.csv"

# Output predictions to a single file
OUTPUT_PREDICTIONS_CSV = "meta_trained_predictions.csv"

# ===== CUSTOMIZATION NOTES =====
# To use with different data:
# 1. Update TRAIN_EVENTS_CSV and ACTUAL_RESULTS_CSV to point to your training data
# 2. Update PREDICT_EVENTS_CSV to point to events you want to predict
# 3. Ensure all model output CSVs (SCENARIO_CSV, etc.) contain predictions from your base models
# 4. Adjust SMOOTHING_K if you have more/less training data (higher K = more smoothing toward 0.5)
# =================================
PROMPT_COL_CANDIDATES = ["interpreted_question", "question"]
ANSWER_OPTIONS_COL = "answer_options"

MODEL_NAME = "gpt-5.2"

MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0

# ===== NEW: multiple passes =====
# Removed - now runs single pass only

# ===== NEW: smoothing for scarce data =====
# Adds k pseudo-trials at 50% accuracy for each model.
# Bigger k => heavier shrink toward 0.5
SMOOTHING_K = 10

# Optional: temperature adds diversity; keep low if you want determinism
TEMPERATURE = 0.2
# =====================================


def call_openai_with_retries(client: OpenAI, messages, model=MODEL_NAME, temperature=TEMPERATURE):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(
                s in msg for s in [
                    "rate limit", "429",
                    "timeout", "timed out",
                    "connection", "connect", "socket",
                    "502", "503", "504",
                    "server error", "bad gateway", "gateway",
                    "temporarily unavailable",
                ]
            )
            if (not transient) or (attempt == MAX_RETRIES):
                raise

            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s *= (0.7 + 0.6 * random.random())
            print(f"[retry {attempt}/{MAX_RETRIES}] transient error: {e} — sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise last_err


def pick_prompt_col(df: pd.DataFrame) -> str:
    for c in PROMPT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"None of {PROMPT_COL_CANDIDATES} found. Columns={list(df.columns)}")


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    return s


def split_answer_options(options_str: str) -> List[str]:
    return [o.strip() for o in str(options_str or "").split(",") if o.strip()]


def snap_to_option(pred: str, options_str: str) -> str:
    pred = (pred or "").strip()
    opts = split_answer_options(options_str)
    if not pred or not opts:
        return pred

    if pred in opts:
        return pred

    lower_map = {o.lower(): o for o in opts}
    if pred.lower() in lower_map:
        return lower_map[pred.lower()]

    if pred.lower().startswith("answer:"):
        stripped = pred.split(":", 1)[1].strip()
        if stripped in opts:
            return stripped
        if stripped.lower() in lower_map:
            return lower_map[stripped.lower()]

    return pred


def simple_correct(pred: str, actual: str, options_str: str) -> bool:
    if not pred or not actual:
        return False
    pred2 = snap_to_option(pred, options_str)
    return normalize_text(pred2) == normalize_text(actual)


def read_model_predictions(
    path: str,
    *,
    event_tickers_ref: List[str],
    expected_pred_cols: List[str],
    prefer_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)

    pred_col = None
    if prefer_col and prefer_col in cols:
        pred_col = prefer_col
    else:
        for c in expected_pred_cols:
            if c in cols:
                pred_col = c
                break
    if pred_col is None:
        raise ValueError(f"None of prediction columns {expected_pred_cols} found in {path}. Columns={cols}")

    if EVENT_TICKER_COL in cols:
        out = df[[EVENT_TICKER_COL, pred_col]].copy()
        out = out.rename(columns={pred_col: "pred"})
        out[EVENT_TICKER_COL] = out[EVENT_TICKER_COL].fillna("").astype(str)
        out["pred"] = out["pred"].fillna("").astype(str)
        return out

    print(f"WARNING: {path} has no '{EVENT_TICKER_COL}'. Falling back to row-index alignment.")
    preds = df[pred_col].fillna("").astype(str).tolist()
    if len(preds) < len(event_tickers_ref):
        preds = preds + [""] * (len(event_tickers_ref) - len(preds))
    else:
        preds = preds[:len(event_tickers_ref)]
    out = pd.DataFrame({EVENT_TICKER_COL: event_tickers_ref, "pred": preds})
    return out


def build_training_table() -> pd.DataFrame:
    train_df = pd.read_csv(TRAIN_EVENTS_CSV)
    for col in [EVENT_TICKER_COL, ANSWER_OPTIONS_COL]:
        if col not in train_df.columns:
            raise ValueError(f"Missing {col!r} in {TRAIN_EVENTS_CSV}")
    prompt_col = pick_prompt_col(train_df)

    actual_df = pd.read_csv(ACTUAL_RESULTS_CSV)
    if EVENT_TICKER_COL not in actual_df.columns:
        raise ValueError(f"Missing {EVENT_TICKER_COL!r} in {ACTUAL_RESULTS_CSV}")
    if "actual_answer" not in actual_df.columns:
        raise ValueError("Missing 'actual_answer' in actual results CSV")

    base = train_df[[EVENT_TICKER_COL, prompt_col, ANSWER_OPTIONS_COL]].copy()
    base = base.rename(columns={prompt_col: "prompt"})

    actual_keep_cols = [
        EVENT_TICKER_COL,
        "actual_answer",
        "actual_market_ticker",
        "market_result",
        "settlement_ts",
        "market_status",
        "debug_note",
    ]
    actual_keep_cols = [c for c in actual_keep_cols if c in actual_df.columns]
    actual_sub = actual_df[actual_keep_cols].copy()

    merged = base.merge(actual_sub, on=EVENT_TICKER_COL, how="left")
    event_list = merged[EVENT_TICKER_COL].fillna("").astype(str).tolist()

    scenario = read_model_predictions(SCENARIO_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "scenario_guess"})
    pe = read_model_predictions(PROMPT_ENG_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "prompt_engineering_guess"})
    mostpop = read_model_predictions(MOST_POP_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "most_popular_informed_guess"})
    control = read_model_predictions(CONTROL_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "control_model_guess"})
    delphi = read_model_predictions(DELPHI_CSV, event_tickers_ref=event_list, expected_pred_cols=["final_answer"]) \
        .rename(columns={"pred": "delphi_guess"})
    ensemble = read_model_predictions(
        ENSEMBLE5_CSV,
        event_tickers_ref=event_list,
        expected_pred_cols=["ensemble5_output", "model_output"],
        prefer_col="ensemble5_output",
    ).rename(columns={"pred": "ensemble5_guess"})

    out = merged.copy()
    for mdf in [scenario, pe, mostpop, control, delphi, ensemble]:
        out = out.merge(mdf, on=EVENT_TICKER_COL, how="left")

    for col in [
        "scenario_guess",
        "prompt_engineering_guess",
        "most_popular_informed_guess",
        "control_model_guess",
        "delphi_guess",
        "ensemble5_guess",
    ]:
        out[col] = out.apply(lambda r: snap_to_option(str(r.get(col, "") or ""), str(r.get("answer_options", "") or "")), axis=1)

    for col in [
        "scenario_guess",
        "prompt_engineering_guess",
        "most_popular_informed_guess",
        "control_model_guess",
        "delphi_guess",
        "ensemble5_guess",
    ]:
        out[col + "_correct_simple"] = out.apply(
            lambda r: simple_correct(
                str(r.get(col, "") or ""),
                str(r.get("actual_answer", "") or ""),
                str(r.get("answer_options", "") or ""),
            ),
            axis=1,
        )

    return out


def compute_model_accuracies(training_table: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """
    Returns:
      raw_acc dict,
      smoothed_acc dict,
      n_resolved (nonblank actual_answer)
    """
    tt = training_table.copy()
    tt["actual_answer"] = tt["actual_answer"].fillna("").astype(str)
    tt = tt[tt["actual_answer"].str.strip() != ""].copy()
    n_resolved = len(tt)

    mapping = {
        "scenario": "scenario_guess_correct_simple",
        "prompt_engineering": "prompt_engineering_guess_correct_simple",
        "most_popular_informed": "most_popular_informed_guess_correct_simple",
        "control": "control_model_guess_correct_simple",
        "delphi": "delphi_guess_correct_simple",
        "ensemble5": "ensemble5_guess_correct_simple",
    }

    raw = {}
    smoothed = {}
    for k, col in mapping.items():
        if col not in tt.columns or n_resolved == 0:
            raw[k] = 0.0
            smoothed[k] = 0.5
            continue

        p = float(tt[col].mean())
        raw[k] = p

        # smoothing toward 0.5 with SMOOTHING_K pseudo-observations at 0.5
        # Equivalent to: (p*n + 0.5*k) / (n + k)
        smoothed[k] = (p * n_resolved + 0.5 * SMOOTHING_K) / (n_resolved + SMOOTHING_K)

    return raw, smoothed, n_resolved


def build_predict_inputs() -> pd.DataFrame:
    df = pd.read_csv(PREDICT_EVENTS_CSV)
    for col in [EVENT_TICKER_COL, ANSWER_OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing {col!r} in {PREDICT_EVENTS_CSV}")
    prompt_col = pick_prompt_col(df)

    out = df[[EVENT_TICKER_COL, prompt_col, ANSWER_OPTIONS_COL]].copy()
    out = out.rename(columns={prompt_col: "prompt"})
    event_list = out[EVENT_TICKER_COL].fillna("").astype(str).tolist()

    # Use the configured model output paths
    scenario = read_model_predictions(SCENARIO_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "scenario_guess"})
    pe = read_model_predictions(PROMPT_ENG_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "prompt_engineering_guess"})
    mostpop = read_model_predictions(MOST_POP_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "most_popular_informed_guess"})
    control = read_model_predictions(CONTROL_CSV, event_tickers_ref=event_list, expected_pred_cols=["model_output"]) \
        .rename(columns={"pred": "control_model_guess"})
    delphi = read_model_predictions(DELPHI_CSV, event_tickers_ref=event_list, expected_pred_cols=["final_answer"]) \
        .rename(columns={"pred": "delphi_guess"})
    ensemble = read_model_predictions(ENSEMBLE5_CSV, event_tickers_ref=event_list,
                                      expected_pred_cols=["ensemble5_output", "model_output"],
                                      prefer_col="ensemble5_output") \
        .rename(columns={"pred": "ensemble5_guess"})

    for mdf in [scenario, pe, mostpop, control, delphi, ensemble]:
        out = out.merge(mdf, on=EVENT_TICKER_COL, how="left")

    for col in [
        "scenario_guess",
        "prompt_engineering_guess",
        "most_popular_informed_guess",
        "control_model_guess",
        "delphi_guess",
        "ensemble5_guess",
    ]:
        out[col] = out.apply(lambda r: snap_to_option(str(r.get(col, "") or ""), str(r.get("answer_options", "") or "")), axis=1)

    return out


def parse_two_line_prediction(text: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "", ""
    ans = lines[0].strip()
    expl = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    return ans, expl


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY2")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY2 not found in environment/.env")
    client = OpenAI(api_key=api_key)

    # 1) Training table
    training_table = build_training_table()
    training_table.to_csv(OUT_TRAINING_TABLE_CSV, index=False)
    print(f"Saved training table: {OUT_TRAINING_TABLE_CSV} (rows={len(training_table)})")

    # 2) Accuracies (raw + smoothed)
    raw_acc, smooth_acc, n_resolved = compute_model_accuracies(training_table)
    print(f"Resolved training rows (nonblank actual_answer): {n_resolved}")
    print("Raw accuracies:", raw_acc)
    print("Smoothed accuracies:", smooth_acc)

    # Store in a compact JSON string to write into prediction rows
    acc_payload = {
        "n_resolved": n_resolved,
        "raw_accuracy": raw_acc,
        "smoothed_accuracy": smooth_acc,
        "smoothing_k": SMOOTHING_K,
    }
    acc_payload_str = json.dumps(acc_payload, separators=(",", ":"))

    # 3) Predict inputs
    predict_df = build_predict_inputs()
    print(f"Loaded predict set: {PREDICT_EVENTS_CSV} (rows={len(predict_df)})")

    SYSTEM_PROMPT = """
You are a meta-forecaster used in a scientific evaluation.
You are given:
- A question, answer options, and several base-model guesses.
- Empirical training-set accuracies for each base model (smoothed due to limited data).

Task:
Choose exactly ONE answer option from ANSWER_OPTIONS.

Rules:
- Prefer answers supported by higher-accuracy models and valid options.
- If a guess is not one of the answer options, discount it.
- Agreement matters, weighted by accuracy.
- Output must be EXACTLY two lines:
  Line 1: the chosen answer option (copied verbatim from ANSWER_OPTIONS)
  Line 2: a brief scientific-style rationale in <= 2 sentences.
"""

    smooth_acc_str = json.dumps(smooth_acc, indent=2)

    # 4) Single prediction pass
    meta_answers = []
    meta_explanations = []

    print(f"\n===== META PREDICTION =====")

    for i, row in predict_df.iterrows():
        event_ticker = str(row.get("event_ticker", "") or "").strip()
        prompt = str(row.get("prompt", "") or "").strip()
        opts = str(row.get("answer_options", "") or "").strip()

        g = {
            "scenario": str(row.get("scenario_guess", "") or "").strip(),
            "prompt_engineering": str(row.get("prompt_engineering_guess", "") or "").strip(),
            "delphi": str(row.get("delphi_guess", "") or "").strip(),
            "most_popular_informed": str(row.get("most_popular_informed_guess", "") or "").strip(),
            "control": str(row.get("control_model_guess", "") or "").strip(),
            "ensemble5": str(row.get("ensemble5_guess", "") or "").strip(),
        }

        print(f"Predicting row {i} ({event_ticker})...")

        user_msg = f"""
TRAINING-SET ACCURACIES (SMOOTHED):
{smooth_acc_str}

EVENT_TICKER: {event_ticker}
QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

BASE MODEL GUESSES:
- scenario: {g["scenario"]}
- prompt_engineering: {g["prompt_engineering"]}
- delphi: {g["delphi"]}
- most_popular_informed: {g["most_popular_informed"]}
- control: {g["control"]}
- ensemble5: {g["ensemble5"]}
"""

        resp = call_openai_with_retries(
            client,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
        raw = resp.choices[0].message.content.strip()
        ans, expl = parse_two_line_prediction(raw)

        ans = snap_to_option(ans, opts)

        meta_answers.append(ans)
        meta_explanations.append(expl)

    out_pred = predict_df.copy()
    out_pred["trained_model_guess"] = meta_answers
    out_pred["brief_explanation"] = meta_explanations

    # Embed the accuracy payload into every row for traceability
    out_pred["meta_training_stats_json"] = acc_payload_str

    out_pred.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
    print(f"Saved predictions: {OUTPUT_PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
