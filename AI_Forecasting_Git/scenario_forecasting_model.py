import pandas as pd
import csv
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("KEY PREFIX:", api_key[:8] if api_key else "MISSING")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment/.env")

client = OpenAI(api_key=api_key)

"""
Scenario Forecasting Model

Implements scenario-based probabilistic forecasting where the AI imagines multiple
future scenarios (optimistic, pessimistic, typical) and translates them into
answer probabilities.

Key methodology:
- AI considers multiple scenarios internally (best-case, worst-case, typical)
- Considers drivers like politics, economics, technology, public opinion
- Translates scenario probabilities into answer option probabilities
- Selects the single most likely answer

Configuration:
- INPUT_CSV: Path to events CSV
- OUTPUT_CSV: Path to save predictions
"""

SCENARIO_SYSTEM_PROMPT = """
    You are an expert probabilistic forecaster. You must:
    1. Read a forecasting QUESTION and its ANSWER_OPTIONS.
    2. Silently think through at least 3 scenarios:
       - An optimistic / best-case scenario.
       - A pessimistic / worst-case scenario.
       - A typical / middle scenario.
       Consider drivers like politics, economics, technology, public opinion, and institutions.
    3. Silently assign probabilities to these scenarios and then translate them into probabilities for each answer option.
    4. Choose the single most likely answer option.

    Important:
    - Do all scenario reasoning internally.
    - DO NOT show your scenario analysis or probabilities in your final output.
"""

EVENT_TICKER_COL = "event_ticker"
PROMPT_COLUMN_NAME = "interpreted_question"
ANSWER_OPTIONS_COL = "answer_options"

INPUT_CSV = "your_events_dataset.csv"
OUTPUT_CSV = "scenario_forecasting_model_outputs.csv"

# ===== CUSTOMIZATION NOTES =====
# To use with different data:
# 1. Update INPUT_CSV to point to your events CSV
# 2. Ensure input CSV has event_ticker, interpreted_question, answer_options columns
# 3. Update OUTPUT_CSV if you want different output filename
# =================================


def safe_request(messages, model="gpt-5.2", max_retries=8):
    """
    Make a chat completion request with basic backoff on 429/rate limit.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            # small sleep after each successful call to avoid hitting TPM limits
            time.sleep(0.25)
            return resp
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str or "429" in err_str:
                wait = 1.5 + attempt * 0.75
                print(f"Rate limit hit (attempt {attempt+1}). Sleeping {wait:.2f}s...")
                time.sleep(wait)
                continue
            # other errors: don't keep retrying
            raise e
    raise RuntimeError("Max retries exceeded due to rate limits.")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_request_with_retries(messages: list, model: str = MODEL_NAME, max_retries: int = MAX_RETRIES_PER_QUESTION) -> str:
    """
    Make OpenAI API request with exponential backoff for rate limits.
    
    Rate limiting is common when making many sequential calls. This function
    catches 429 errors and sleeps with backoff before retrying.
    
    Args:
        messages: List of message dicts with "role" and "content"
        model: Model name (e.g., "gpt-4")
        max_retries: Number of retry attempts before giving up
    
    Returns:
        Response text from model
    
    Raises:
        RuntimeError if max retries exceeded or non-transient error occurs
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            # Transient error: rate limit
            if "rate_limit" in err_str or "429" in err_str:
                wait_time = 1.5 + attempt * 0.75
                print(f"  Rate limit (attempt {attempt+1}/{max_retries}). Sleeping {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            # Non-transient error: fail immediately
            raise RuntimeError(f"API error on attempt {attempt+1}: {e}") from e
    
    raise RuntimeError(f"Max retries ({max_retries}) exceeded due to persistent rate limiting.")


def parse_response(text: str) -> tuple[str, str]:
    """
    Parse model output into answer and optional explanation.
    
    Expected format (model should output ~2 lines):
      Line 1: Selected answer option
      Line 2+: Optional brief explanation
    
    Args:
        text: Raw model response text
    
    Returns:
        Tuple of (answer, explanation)
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "", ""
    answer = lines[0]
    explanation = " ".join(lines[1:]) if len(lines) > 1 else ""
    return answer, explanation


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================

def run_scenario_forecasting(input_csv: str, output_csv: str):
    """
    Run scenario-based forecasting on a dataset.
    
    Uses scenario reasoning where the AI imagines multiple future scenarios
    (optimistic, pessimistic, typical) and translates them into answer probabilities.
    
    Args:
        input_csv: Path to events CSV with columns: event_ticker, interpreted_question, answer_options
        output_csv: Path to save predictions
    
    Raises:
        FileNotFoundError: If input CSV not found
        ValueError: If required columns missing
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} events from {input_csv}")
    
    # Validate required columns
    for col in [EVENT_TICKER_COL, PROMPT_COLUMN_NAME, ANSWER_OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")
    
    event_tickers = df[EVENT_TICKER_COL].fillna("").astype(str).tolist()
    prompts = df[PROMPT_COLUMN_NAME].fillna("").astype(str).tolist()
    options_per_row = df[ANSWER_OPTIONS_COL].fillna("").astype(str).tolist()
    
    print(f"\n{'='*70}")
    print("RUNNING SCENARIO FORECASTING")
    print(f"{'='*70}")
    
    chosen_answers = []
    explanations = []
    
    for i, (ticker, prompt, opts) in enumerate(zip(event_tickers, prompts, options_per_row)):
        if i % 50 == 0:
            print(f"  [{i}/{len(event_tickers)}] Processing events...")
        
        try:
            user_msg = f"""
QUESTION: {prompt}
ANSWER_OPTIONS: {opts}

Using scenario-based reasoning:
1. Imagine multiple scenarios (optimistic, pessimistic, typical, etc.)
2. Consider key factors (political, economic, technological, social)
3. Estimate scenario probabilities and translate to answer options
4. Select the most likely option

Output format (two lines):
Line 1: Your selected answer (must be from ANSWER_OPTIONS)
Line 2: Brief justification (<= 2 sentences, describe your key scenarios)
"""
            
            response_text = safe_request_with_retries(
                messages=[
                    {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                model=MODEL_NAME
            )
            
            answer, explanation = parse_response(response_text)
            chosen_answers.append(answer)
            explanations.append(explanation)
            
        except Exception as e:
            print(f"  ERROR on event {i} ({ticker}): {e}")
            chosen_answers.append(f"ERROR")
            explanations.append(str(e))
    
    # Save results
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_ticker", "prompt", "answer_options", "model_output", "explanation"])
        for ticker, prompt, opts, ans, expl in zip(
            event_tickers, prompts, options_per_row, chosen_answers, explanations
        ):
            writer.writerow([ticker, prompt, opts, ans, expl])
    
    print(f"✓ Saved predictions to: {output_csv}")
    print(f"  Correct: {sum(1 for a in chosen_answers if not a.startswith('ERROR'))}/{len(chosen_answers)}")


def main():
    """
    Run scenario forecasting on the configured dataset.
    """
    run_scenario_forecasting(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()

