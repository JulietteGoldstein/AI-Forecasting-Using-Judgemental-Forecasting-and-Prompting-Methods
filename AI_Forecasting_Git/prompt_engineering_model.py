"""
Prompt Engineering Forecasting Model

This module implements a two-stage prompt engineering approach for probabilistic forecasting:
1. **Stage 1 (Engineering)**: Uses an AI to design optimized instructions for a forecasting task
2. **Stage 2 (Forecasting)**: Uses the engineered prompt to make predictions

The model demonstrates that well-designed prompts can improve AI forecasting accuracy
by breaking down complex reasoning and providing structured output requirements.

Key references:
- Prompt brittleness: Small wording changes lead to different outputs
- Decomposition: Breaking tasks into subtasks improves performance
- Format specificity: Clear output requirements stabilize responses

Configuration:
- INPUT_CSV: Path to events CSV with columns [event_ticker, interpreted_question, answer_options]
- OUTPUT_CSV: Path to save model predictions
- SLEEP_SECONDS: Delay between API calls (prevents rate limiting)
- MAX_RETRIES: Number of retry attempts for transient API errors
"""

import pandas as pd
import csv
import os
import time
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("KEY PREFIX:", api_key[:8] if api_key else "MISSING")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment/.env")

client = OpenAI(api_key=api_key)

# ===== CONFIGURATION =====
INPUT_CSV = "kalshi_collect.csv"
OUTPUT_CSV = "prompt_engineering_forecasting_outputs.csv"

# ===== RATE LIMITING / RELIABILITY =====
SLEEP_SECONDS = 0.6          # base sleep between successful calls
MAX_RETRIES = 8              # retries per call
BACKOFF_BASE_SECONDS = 1.0   # exponential backoff base
BACKOFF_MAX_SECONDS = 30.0   # cap
# ======================================

PROMPT_ENGINEERING_NOTES = """
Self prompt engineering:
- Self-prompt engineering drives AI-generated design and iteration of instructions for an AI.
- Effective prompt engineering often involves decomposing complex tasks into smaller subtasks or stages.
- Prompts that have too rigid instructions can limit creativity, while too vague prompts lead to incoherent output;
  adding examples or constraints helps stabilize responses.
- Prompt brittleness: small changes in wording or order of instructions can lead to wildly different outputs,
  so prompt testing and iteration are critical to robustness.
-Engineered prompts are better when the person making them has researched the topic and can therefore give contextual information on recent events or trends regarding the subject. 

Forecasting & LLM behavior:
- LLMs are initially better at predicting the signs of estimated effects than making estimates themselves.
- LLMs work effectively when acting as human characters, making predictions based on what those humans would do.
- LLMs can improve forecasts with preprompting alone.
- LLMs still struggle with specific instructions, even when preprompted — "show, not tell" often works better.

Human + AI ensembles:
- LLMs benefit from human prediction as a basis.
- Human forecasting (wisdom of the crowd) generally outperforms standalone AI forecasts.
- LLMs in an ensemble can become on par with human forecasts.
- Supplying the LLMs with the median human forecast boosts accuracy.

Simulation & experimentation:
- Simulated AI agents are low cost, fast, scalable, and accurate enough to be on par with human scenarios.
- There's no need to consider human-subject ethics for these artificial agents.
- For AI experiments, pre-registration is less important because runs are cheap and fast.

AI in forecasting complex systems:
- Conventional methods depend heavily on historical data and may fail in volatile environments.
- AI can integrate variables like sentiment, social media, weather, and macro indicators to adapt to shifts.
- AI accuracy depends on large, accurate, structured datasets; data silos and messy data reduce reliability.
- Larger organizations can better overcome data challenges than smaller firms.
- Upfront costs (software, integration, talent) are nontrivial, especially for smaller firms.
- Long-term benefits include efficiency, cost savings, and competitive advantage.
"""

EVENT_TICKER_COL = "event_ticker"
PROMPT_COLUMN_NAME = "interpreted_question"
ANSWER_OPTIONS_COL = "answer_options"

# ===== CUSTOMIZATION NOTES =====
# To use with a different dataset:
# 1. Replace INPUT_CSV with path to your events file
# 2. Ensure your CSV has columns: event_ticker, interpreted_question, answer_options
# 3. Adjust SLEEP_SECONDS if hitting rate limits (increase) or if running too slowly (decrease)
# 4. Increase MAX_RETRIES for unreliable network conditions
# =================================


def call_openai_with_retries(messages, model="gpt-5.2"):
    """
    Retries on transient failures (rate limits, timeouts, connection resets, 5xx).
    Keeps the rest of your code structure the same.
    """
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return resp

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # treat these as transient
            transient = any(
                s in msg
                for s in [
                    "rate limit",
                    "429",
                    "timeout",
                    "timed out",
                    "connection",
                    "connect",
                    "socket",
                    "502",
                    "503",
                    "504",
                    "server error",
                    "bad gateway",
                    "gateway",
                    "temporarily unavailable",
                ]
            )

            if not transient or attempt == MAX_RETRIES:
                raise

            # exponential backoff with jitter
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.7 + 0.6 * random.random())  # jitter
            print(f"[retry {attempt}/{MAX_RETRIES}] transient error: {e} — sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)

    # should never reach here
    raise last_err


def run_one(input_csv_path: str, out_path: str):
    """
    Run prompt engineering forecasting on a single dataset.
    
    Args:
        input_csv_path: Path to input CSV
        out_path: Path to save output CSV
    """
    df = pd.read_csv(input_csv_path)
    print("Columns:", list(df.columns))

    for col in [EVENT_TICKER_COL, PROMPT_COLUMN_NAME, ANSWER_OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {input_csv_path}")

    event_tickers = df[EVENT_TICKER_COL].fillna("").tolist()
    prompts = df[PROMPT_COLUMN_NAME].fillna("").tolist()
    options_per_row = df[ANSWER_OPTIONS_COL].fillna("").tolist()

    engineered_prompts = []
    final_answers = []

    for i, (prompt, opts) in enumerate(zip(prompts, options_per_row)):
        print(f"\n=== Row {i} ===")
        print(f"EVENT_TICKER: {event_tickers[i]}")
        print(f"QUESTION: {prompt}")
        print(f"ANSWER_OPTIONS: {opts}")

        # ---------- SESSION 1: engineer prompt ----------
        try:
            pe_user_msg = f"""
You are an expert prompt engineer and forecasting methodologist.

Below are some notes about prompt engineering, forecasting, scenario reasoning,
human–AI ensembles, and tool-augmented reasoning:

{PROMPT_ENGINEERING_NOTES}

Design an *instruction prompt* that will be given to another AI model to make a probabilistic forecast and then select ONE answer option.

   Requirements for the engineered instruction prompt:
   1. The specific listed instructions should:
      - Decompose the question-specific reasoning process into smaller logical steps.
      - Use outside real-world knowledge, but DO NOT actually call tools or APIs.
      - Not be vague; they should be specific to the prompt at hand.
   2. It should warn about prompt brittleness and emphasize following the format strictly.
   3. It must specify final visible output format:
      <exactly ONE of the provided answer options, copied verbatim, with no additional words or labels>
   4. The engineered prompt should NOT answer the question itself.

QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Respond with ONLY the engineered instruction prompt text.
"""
            pe_resp = call_openai_with_retries(
                messages=[
                    {"role": "system", "content": "You interpret messy forecasting questions and design high-quality prompts for other AI models."},
                    {"role": "user", "content": pe_user_msg},
                ],
                model="gpt-5.2",
            )

            engineered = pe_resp.choices[0].message.content.strip()
            engineered_prompts.append(engineered)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"Error in Session 1 on row {i}: {e}")
            engineered = f"ERROR_ENGINEERING: {e}"
            engineered_prompts.append(engineered)

        # ---------- SESSION 2: forecast using engineered prompt ----------
        try:
            forecasting_user_msg = f"""
You are the forecasting model.

QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Follow the instructions given in the system prompt.

Your visible reply MUST be:
- Exactly one of the provided answer options,
- Copied verbatim,
- With no extra words, no labels, no explanations.
Just the answer option text.
"""
            resp = call_openai_with_retries(
                messages=[
                    {"role": "system", "content": engineered},
                    {"role": "user", "content": forecasting_user_msg},
                ],
                model="gpt-5.2",
            )

            raw = resp.choices[0].message.content.strip()
            final_answers.append(raw)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"Error forecasting on row {i}: {e}")
            final_answers.append(f"ERROR_FORECAST: {e}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_ticker", "prompt", "answer_options", "engineered_prompt", "model_output"])
        for ev, p, opts, eng, ans in zip(event_tickers, prompts, options_per_row, engineered_prompts, final_answers):
            writer.writerow([ev, p, opts, eng, ans])

    print(f"\nDone! Saved outputs to: {out_path}")


def main():
    """Run prompt engineering forecasting on the configured dataset."""
    run_one(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
