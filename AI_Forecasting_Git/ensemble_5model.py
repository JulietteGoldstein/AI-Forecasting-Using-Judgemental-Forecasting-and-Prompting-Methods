"""
5-Model Ensemble Forecasting

This module creates an ensemble forecast by combining predictions from 5 different forecasting models:
1. Scenario forecasting
2. Prompt engineering
3. Delphi panel
4. Most-popular-option informed
5. Control baseline

The ensemble uses meta-reasoning: it presents all 5 predictions to the AI and asks it to choose
the single most likely answer, which can improve accuracy through ensemble methods.

Configuration:
- INPUT_CSV: Path to events CSV (base data)
- SCENARIO_CSV, PE_CSV, etc.: Individual model outputs to combine
- OUTPUT_CSV: Path to save ensemble predictions
"""

import pandas as pd
import csv
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)

# ===== CONFIGURATION =====
INPUT_CSV = "your_events_dataset.csv"
SCENARIO_CSV = "scenario_forecasting_model_outputs.csv"
PE_CSV = "prompt_engineering_forecasting_outputs.csv"
MOSTPOP_CSV = "most_popular_option_model_outputs.csv"
CONTROL_CSV = "control_model_outputs.csv"
DELPHI_CSV = "delphi_outputs.csv"
OUTPUT_CSV = "ensemble5_model_outputs.csv"

EVENT_TICKER_COL = "event_ticker"
PROMPT_COL = "interpreted_question"
OPTIONS_COL = "answer_options"

# ===== CUSTOMIZATION NOTES =====
# To use with different data:
# 1. Update all the *_CSV paths to point to your model outputs
# 2. Ensure INPUT_CSV has event_ticker, interpreted_question, answer_options
# 3. Ensure each model output CSV has event_ticker and model_output columns
# =================================

# ========= HELPER FUNCTIONS =========

def load_map(path: str, key_col: str, val_col: str):
    """
    Load a mapping key_col -> val_col from CSV.
    If duplicates exist, keep the *last non-empty* value.
    """
    df = pd.read_csv(path)
    if key_col not in df.columns:
        raise ValueError(f"Column {key_col!r} not found in {path}")
    if val_col not in df.columns:
        raise ValueError(f"Column {val_col!r} not found in {path}")

    m = {}
    for _, r in df.iterrows():
        k = str(r.get(key_col, ""))
        v = str(r.get(val_col, "") if pd.notna(r.get(val_col, "")) else "")
        if v.strip() != "":
            m[k] = v
        else:
            # only set empty if key not already set (so we don't overwrite good values)
            if k not in m:
                m[k] = ""
    return m


def run_ensemble(input_csv: str, scenario_csv: str, pe_csv: str, delphi_csv: str,
                 mostpop_csv: str, control_csv: str, output_csv: str):
    """
    Run 5-model ensemble forecasting.
    
    Args:
        input_csv: Base events CSV
        scenario_csv, pe_csv, delphi_csv, mostpop_csv, control_csv: Individual model outputs
        output_csv: Path to save ensemble predictions
    """
    df = pd.read_csv(input_csv)

    for col in [EVENT_TICKER_COL, PROMPT_COL, OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {input_csv}")

    # Load guess maps keyed by event_ticker
    scenario_map = load_map(scenario_csv, EVENT_TICKER_COL, "model_output")
    pe_map = load_map(pe_csv, EVENT_TICKER_COL, "model_output")
    mostpop_map = load_map(mostpop_csv, EVENT_TICKER_COL, "model_output")
    control_map = load_map(control_csv, EVENT_TICKER_COL, "model_output")
    delphi_map = load_map(delphi_csv, EVENT_TICKER_COL, "final_answer")

    ensemble_outputs = []

    for i, row in df.iterrows():
        ev = str(row.get(EVENT_TICKER_COL, ""))
        prompt = str(row.get(PROMPT_COL, ""))
        opts = str(row.get(OPTIONS_COL, ""))

        g_scenario = scenario_map.get(ev, "")
        g_pe = pe_map.get(ev, "")
        g_delphi = delphi_map.get(ev, "")
        g_mostpop = mostpop_map.get(ev, "")
        g_control = control_map.get(ev, "")

        print(f"Ensembling row {i} ({ev})...")

        msg = f"""
QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Predictions from five different methods:

1) Scenario forecasting: {g_scenario}
2) Prompt-engineering model: {g_pe}
3) Delphi panel final answer: {g_delphi}
4) Most-popular-informed model: {g_mostpop}
5) Control model: {g_control}

Choose exactly ONE answer option from ANSWER_OPTIONS.
Respond with ONLY the answer option text, copied verbatim.
"""

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": msg}],
        )

        ensemble_outputs.append(resp.choices[0].message.content.strip())

    # Write output CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_ticker",
            "prompt",
            "answer_options",
            "scenario_guess",
            "prompt_engineering_guess",
            "delphi_guess",
            "most_popular_informed_guess",
            "control_model_guess",
            "ensemble5_output",
        ])

        for i, row in df.iterrows():
            ev = str(row.get(EVENT_TICKER_COL, ""))
            prompt = str(row.get(PROMPT_COL, ""))
            opts = str(row.get(OPTIONS_COL, ""))

            writer.writerow([
                ev,
                prompt,
                opts,
                scenario_map.get(ev, ""),
                pe_map.get(ev, ""),
                delphi_map.get(ev, ""),
                mostpop_map.get(ev, ""),
                control_map.get(ev, ""),
                ensemble_outputs[i],
            ])

    print(f"Saved ensemble to {output_csv}")

def main():
    """Run 5-model ensemble on the configured dataset."""
    run_ensemble(
        INPUT_CSV,
        SCENARIO_CSV,
        PE_CSV,
        DELPHI_CSV,
        MOSTPOP_CSV,
        CONTROL_CSV,
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()
