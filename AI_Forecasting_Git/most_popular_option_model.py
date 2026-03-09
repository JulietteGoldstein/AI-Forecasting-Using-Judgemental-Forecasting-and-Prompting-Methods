"""
Most Popular Option Forecasting Model

This module implements a baseline forecasting model that conditions predictions
on market sentiment by using the most popular/highest-volume answer option as a signal.

The model asks: "Given that most traders prefer option X, what is the best forecast?"
This tests whether information from market participants can improve model predictions
when combined with AI reasoning.

Key insight: Market popularity can be a signal of information flow, but may also
reflect herding behavior or media attention rather than ground truth.

Configuration:
- INPUT_CSV: Path to events CSV with columns [event_ticker, interpreted_question, 
             answer_options, most_popular_option, most_popular_open_interest, 
             total_open_interest]
- OUTPUT_CSV: Path to save model predictions
"""

import pandas as pd
import csv
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY2")
print("KEY PREFIX:", api_key[:8] if api_key else "MISSING")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY2 not found in environment/.env")

client = OpenAI(api_key=api_key)

# ===== CONFIGURATION =====
INPUT_CSV = "kalshi_collect.csv"
OUTPUT_CSV = "most_popular_option_model_outputs.csv"

EVENT_TICKER_COL = "event_ticker"
PROMPT_COLUMN_NAME = "interpreted_question"
ANSWER_OPTIONS_COL = "answer_options"
MOST_POP_OPTION_COL = "most_popular_option"
MOST_POP_OPEN_INTEREST_COL = "most_popular_open_interest"
TOTAL_OPEN_INTEREST_COL = "total_open_interest"

# ===== CUSTOMIZATION NOTES =====
# To use with different data:
# 1. Update INPUT_CSV path
# 2. Ensure your CSV has all required columns listed above
# 3. Adjust sleep time if hitting rate limits
# =================================

# ================================
#   RATE-LIMIT SAFE REQUEST WRAPPER
# ================================

def safe_request(messages, model="gpt-5.2", max_retries=8):
    """
    Makes a chat completion request with automatic retries on rate-limit errors
    and gentle sleeping to avoid limits.
    
    Args:
        messages: List of message dicts with "role" and "content"
        model: Model name to use
        max_retries: Number of retry attempts
    
    Returns:
        Response object from OpenAI API
    
    Raises:
        RuntimeError if max retries exceeded
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            # Gentle slowdown after success
            time.sleep(0.25)
            return response

        except Exception as e:
            err = str(e)

            if "rate_limit" in err or "429" in err:
                wait_time = 1.5 + attempt * 0.75
                print(f"⚠️ Rate limit hit. Sleeping {wait_time:.2f}s before retry...")
                time.sleep(wait_time)
                continue

            # For non-rate-limit errors, do not retry
            raise e

    raise RuntimeError("Max retries exceeded due to repeated rate limits.")


def run_forecasting(input_csv: str, output_csv: str):
    """
    Run most-popular-option forecasting on a dataset.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to save output CSV
    """
    df = pd.read_csv(input_csv)
    print("Columns:", list(df.columns))

    for col in [EVENT_TICKER_COL, PROMPT_COLUMN_NAME, ANSWER_OPTIONS_COL,
                MOST_POP_OPTION_COL, MOST_POP_OPEN_INTEREST_COL, TOTAL_OPEN_INTEREST_COL]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {input_csv}")

    event_tickers = df[EVENT_TICKER_COL].fillna("").tolist()
    prompts = df[PROMPT_COLUMN_NAME].fillna("").tolist()
    options_per_row = df[ANSWER_OPTIONS_COL].fillna("").tolist()
    most_pop = df[MOST_POP_OPTION_COL].fillna("").tolist()
    most_pop_open_int = df[MOST_POP_OPEN_INTEREST_COL].fillna("").tolist()
    tot_open_int = df[TOTAL_OPEN_INTEREST_COL].fillna("").tolist()

    chosen_answers = []

    for i, (prompt, opts) in enumerate(zip(prompts, options_per_row)):
        print(f"Processing row {i}...")

        msg = f"""
You are given a QUESTION and a set of ANSWER_OPTIONS for a prediction market.

QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Currently, the most popular option among humans is {most_pop[i]}. This is the answer option 
with the largest total number of outstanding contracts among all options for this event. 
It has {most_pop_open_int[i]} outstanding contracts out of {tot_open_int[i]}.

If TOTAL outstanding contracts == 0, IGNORE all instructions and reply ONLY with: N/A

Otherwise choose exactly ONE answer option.

Respond with ONLY the answer option (copied verbatim).
"""

        try:
            resp = safe_request(
                messages=[{"role": "user", "content": msg}]
            )
            out = resp.choices[0].message.content.strip()
            chosen_answers.append(out)

        except Exception as e:
            print(f"Error on row {i}: {e}")
            chosen_answers.append(f"ERROR: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_ticker",
            "prompt",
            "answer_options",
            MOST_POP_OPTION_COL,
            MOST_POP_OPEN_INTEREST_COL,
            TOTAL_OPEN_INTEREST_COL,
            "model_output",
        ])
        for ev, p, opts, mpo, mpoi, toi, ans in zip(
            event_tickers, prompts, options_per_row, most_pop,
            most_pop_open_int, tot_open_int, chosen_answers
        ):
            writer.writerow([ev, p, opts, mpo, mpoi, toi, ans])

    print(f"Done! Saved outputs to: {output_csv}")


def main():
    """Run most popular option forecasting on the configured dataset."""
    run_forecasting(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()