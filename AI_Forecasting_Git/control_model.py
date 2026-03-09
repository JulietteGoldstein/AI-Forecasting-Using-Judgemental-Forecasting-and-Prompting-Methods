"""
Control Model Forecasting

This module implements a simple baseline forecasting model for comparison purposes.
The control model asks the AI to make a direct forecast without any special prompting,
engineered instructions, or auxiliary information (unlike the other models).

Purpose: Establish a baseline to measure the value of more sophisticated approaches
(scenario reasoning, prompt engineering, market sentiment, etc.).

Configuration:
- INPUT_CSV: Path to events CSV with columns [event_ticker, interpreted_question, answer_options]
- OUTPUT_CSV: Path to save model predictions
"""

import pandas as pd
import csv
import os
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
OUTPUT_CSV = "control_model_outputs.csv"

EVENT_TICKER_COL = "event_ticker"
PROMPT_COLUMN_NAME = "interpreted_question"
ANSWER_OPTIONS_COL = "answer_options"

# ===== CUSTOMIZATION NOTES =====
# To use with different data:
# 1. Update INPUT_CSV and OUTPUT_CSV paths
# 2. Ensure CSV has required columns
# =================================


def run_control_forecasting(input_csv: str, output_csv: str):
    """
    Run control model forecasting on a dataset.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to save output CSV
    """
    df = pd.read_csv(input_csv)
    print("Columns:", list(df.columns))

    for col in [EVENT_TICKER_COL, PROMPT_COLUMN_NAME, ANSWER_OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {input_csv}")

    event_tickers = df[EVENT_TICKER_COL].fillna("").tolist()
    prompts = df[PROMPT_COLUMN_NAME].fillna("").tolist()
    options_per_row = df[ANSWER_OPTIONS_COL].fillna("").tolist()

    chosen_answers = []

    for i, (prompt, opts) in enumerate(zip(prompts, options_per_row)):
        print(f"Processing row {i}...")
        try:
            msg = f"""
You are given a QUESTION and a set of ANSWER_OPTIONS for a prediction market.

QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Choose exactly ONE answer from ANSWER_OPTIONS that best answers the true question.

Respond in exactly this format below (no extra words):

<one of the answer options, copied verbatim>
"""
            resp = client.chat.completions.create(
                model="gpt-5.2",
                messages=[{"role": "user", "content": msg}],
            )
            chosen_answers.append(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error on row {i}: {e}")
            chosen_answers.append(f"ERROR: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_ticker", "prompt", "answer_options", "model_output"])
        for ev, p, opts, ans in zip(event_tickers, prompts, options_per_row, chosen_answers):
            writer.writerow([ev, p, opts, ans])

    print(f"Done! Saved outputs to: {output_csv}")


def main():
    """Run control model forecasting on the configured dataset."""
    run_control_forecasting(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
