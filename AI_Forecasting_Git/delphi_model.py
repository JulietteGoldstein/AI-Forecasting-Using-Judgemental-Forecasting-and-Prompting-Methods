"""
Delphi Forecasting Model

Implements an AI-based Delphi method for probabilistic forecasting.
Multiple expert personas engage in iterative rounds, with feedback between rounds
to converge on consensus predictions while preserving diverse viewpoints.

Key methodology:
- Round 1: Experts make independent predictions
- Feedback: Show each expert the group distribution and rationales
- Round 2+: Experts revise based on feedback
- Final: Aggregate predictions after convergence

Configuration:
- INPUT_CSV: Path to events CSV
- OUTPUT_CSV: Path to save predictions
- NUM_EXPERTS: Number of distinct expert personas (default: 5)
- NUM_ROUNDS: Number of Delphi rounds (default: 3)
"""

import os
import csv
import time
import random
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "kalshi_collect.csv"
OUTPUT_CSV = "delphi_outputs.csv"

NUM_EXPERTS = 5
NUM_ROUNDS = 3

EVENT_TICKER_COL = "event_ticker"
QUESTION_COL = "interpreted_question"
ANSWER_OPTIONS_COL = "answer_options"

# Reliability settings
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
PARALLEL_WORKERS = min(10, NUM_EXPERTS)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY2")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY2 not found in environment/.env")

client = OpenAI(api_key=api_key)

# Original research prompt - do not modify
DELPHI_NOTES = """
You are designing a Delphi-style forecasting panel using AI experts.

Key points about the Delphi method and prompt engineering:

- A Policy Delphi explores disagreements and different viewpoints, not just consensus.
- eDelphi is online Delphi: fast, cheap, scalable. AI can go beyond human limits (e.g., 20 human experts vs 40–60 AI experts).
- In Delphi, less accurate participants benefit the most from group feedback.
- Bias is a concern; diversity of perspectives helps.
- Delphi is useful when numerical data is scarce, and when questions are complex or controversial.
- A facilitator (here: you) organizes experts, feedback, and iterations.

Prompt engineering notes:
- Break complex tasks into smaller reasoning steps.
- Too rigid prompts kill creativity; too vague prompts cause incoherence. Use examples and constraints.
- Prompts are brittle; wording order matters. Be explicit about required formats.
- Using personas (distinct roles with clear bios) helps LLMs behave like differentiated human experts.
- Hybrid, literature-based prompts lead to more systematic and less random AI behavior.
- Iterative cycles (like Delphi rounds) improve stability and accuracy.

Your job:
- Design a DIVERSE panel of DOMAIN-RELEVANT expert personas for your given question.
- Each expert must have a very topic-specific profession. Never say "generic forecaster".
- Write short, concrete bios that detail how each persona has a unique, useful perspective on the forecast.
"""


def extract_json_array(text: str) -> str:
    t = (text or "").strip()

    # Remove ```json ... ``` or ``` ... ``` wrappers
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    # Extract first JSON array if extra text exists
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]

    return t


def topic_specific_fallback_personas(question: str, n: int):
    """
    If the model truly fails, avoid 'General forecaster' and still produce diverse,
    topic-ish personas based on the question text.
    """
    q = (question or "").lower()

    if any(k in q for k in ["election", "president", "congress", "senate", "vote", "campaign", "trump", "biden", "politic"]):
        base = [
            ("Campaign Strategist", "U.S. political strategist", "Focuses on incentives, coalition dynamics, and electoral fundamentals."),
            ("Pollster", "Survey researcher", "Prioritizes polling trends, likely-voter screens, and historical error patterns."),
            ("Congressional Aide", "Legislative staffer", "Tracks committee dynamics, whip counts, and legislative feasibility."),
            ("Political Journalist", "Capitol Hill reporter", "Weights credible reporting, insider signals, and agenda-setting."),
            ("Constitutional Lawyer", "Election & constitutional attorney", "Focuses on legal constraints, procedural hurdles, and litigation risk."),
        ]
    elif any(k in q for k in ["interest rate", "fed", "inflation", "gdp", "recession", "unemployment", "bond", "treasury", "rates"]):
        base = [
            ("Macro Economist", "Macroeconomist", "Focuses on inflation, growth, and labor-market dynamics."),
            ("Fed Watcher", "Monetary policy analyst", "Emphasizes FOMC reaction function and forward guidance."),
            ("Rates Trader", "Fixed income trader", "Uses market-implied paths, term premia, and positioning."),
            ("Credit Analyst", "Credit markets analyst", "Tracks financial conditions, spreads, and default risk."),
            ("Energy/Commodities Analyst", "Commodities strategist", "Watches supply shocks that feed into inflation."),
        ]
    elif any(k in q for k in ["tesla", "robotaxi", "ai", "model", "chip", "openai", "launch", "tech", "software"]):
        base = [
            ("Tech Product Lead", "Product leader", "Focuses on timelines, incentives, and execution risk."),
            ("Regulatory Analyst", "Transportation/regulatory analyst", "Focuses on approvals, safety, and compliance hurdles."),
            ("Industry Analyst", "Tech industry analyst", "Uses competitor benchmarks and supply-chain signals."),
            ("Engineer", "Senior software/ML engineer", "Focuses on technical feasibility and integration risk."),
            ("Venture Investor", "Tech investor", "Focuses on incentives, capital allocation, and go-to-market patterns."),
        ]
    else:
        base = [
            ("Domain Analyst", "Sector analyst", "Focuses on sector-specific indicators and precedent."),
            ("Risk Manager", "Risk & uncertainty specialist", "Focuses on tail risks and base rates."),
            ("Investigative Journalist", "Research journalist", "Focuses on credible reporting and verification."),
            ("Statistician", "Applied statistician", "Focuses on historical frequency and calibration."),
            ("Policy Analyst", "Policy analyst", "Focuses on institutional constraints and incentives."),
        ]

    out = []
    for i in range(n):
        name, role, bio = base[i % len(base)]
        out.append({"name": name, "role": role, "bio": bio})
    return out


def generate_expert_personas(question: str, answer_options: str, num_experts: int):
    user_msg = f"""
Below are notes on the Delphi method and prompt engineering:

{DELPHI_NOTES}

Now you will design a panel of AI experts for a prediction market question.

QUESTION: {question}
ANSWER_OPTIONS: {answer_options}

Task:
1. Choose {num_experts} distinct expert personas who would bring genuinely different,
   relevant perspectives.
2. For each persona, specify:
   - name
   - role
   - bio

Hard constraints:
- No persona may be "General forecaster" / "Generic forecaster".
- Roles must be topic-specific to this question.

Format:
Return ONLY valid JSON with this exact structure:

[
  {{"name": "...", "role": "...", "bio": "..."}},
  ...
]
"""

    resp = call_openai_with_retries(
        messages=[
            {"role": "system", "content": "You are an expert Delphi facilitator and AI prompt engineer."},
            {"role": "user", "content": user_msg},
        ],
        model="gpt-5.2",
    )
    raw = resp.choices[0].message.content.strip()

    def parse_personas(raw_text: str):
        cleaned_raw = extract_json_array(raw_text)
        personas = json.loads(cleaned_raw)
        cleaned = []
        for p in personas:
            cleaned.append({
                "name": str(p.get("name", "")).strip() or "Expert",
                "role": str(p.get("role", "")).strip() or "Analyst",
                "bio": str(p.get("bio", "")).strip() or "No bio provided."
            })
        return cleaned[:num_experts]

    try:
        return parse_personas(raw)

    except Exception:
        repair_msg = f"""
Your previous response was not valid JSON or had extra text.
Return ONLY valid JSON (no code fences, no commentary), exactly this structure:

[
  {{"name": "...", "role": "...", "bio": "..."}},
  ...
]

You must return {num_experts} items.

QUESTION: {question}
ANSWER_OPTIONS: {answer_options}
"""
        try:
            resp2 = call_openai_with_retries(
                messages=[
                    {"role": "system", "content": "Return strictly valid JSON only."},
                    {"role": "user", "content": repair_msg},
                ],
                model="gpt-5.2",
            )
            raw2 = resp2.choices[0].message.content.strip()
            return parse_personas(raw2)
        except Exception:
            return topic_specific_fallback_personas(question, num_experts)


def format_answer_options_for_prompt(answer_options: str) -> str:
    opts = [o.strip() for o in str(answer_options).split(",") if o.strip()]
    return ", ".join(opts) if opts else answer_options


def run_expert_round(persona, question, answer_options, round_index, previous_answer, all_prev_answers):
    name = persona["name"]
    role = persona["role"]
    bio = persona["bio"]
    opts_formatted = format_answer_options_for_prompt(answer_options)

    crowd_info_str = ""
    if all_prev_answers:
        crowd_info_str = "\n".join([f"{exp_name}: {ans}" for exp_name, ans in all_prev_answers])

    if round_index == 0:
        round_desc = "INITIAL ROUND (no knowledge of other experts yet)"
        instructions = "Make your own forecast independently."
    else:
        round_desc = f"DELPHI UPDATE ROUND {round_index}"
        instructions = """
You are now in a later Delphi round. You have seen the previous round's answers from all experts.
You MUST take them into account.
"""

    user_msg = f"""
You are taking part in a Delphi-style forecasting exercise.

Persona:
- Name: {name}
- Role: {role}
- Bio: {bio}

QUESTION:
{question}

ANSWER_OPTIONS (choose exactly one of these):
{opts_formatted}

ROUND INFO:
{round_desc}

Your previous answer: {previous_answer if previous_answer else "None"}.

All experts' previous answers:
{crowd_info_str if crowd_info_str else "None (first round)."}

{instructions}

Output format (exactly):
ANSWER: <one answer option copied verbatim>
"""

    resp = call_openai_with_retries(
        messages=[
            {"role": "system", "content": "You are a careful, honest probabilistic forecaster. Follow the required output format."},
            {"role": "user", "content": user_msg},
        ],
        model="gpt-5.2",
    )
    raw = resp.choices[0].message.content.strip()

    answer = raw
    for line in raw.splitlines():
        s = line.strip()
        if s.lower().startswith("answer:"):
            answer = s.split(":", 1)[1].strip()
            break
    return answer


def run_delphi(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    for col in [EVENT_TICKER_COL, QUESTION_COL, ANSWER_OPTIONS_COL]:
        if col not in df.columns:
            raise ValueError(f"{col!r} column not found in {input_csv}")

    rows_out = []

    for idx, row in df.iterrows():
        event_ticker = row.get(EVENT_TICKER_COL, "")
        original_question = row.get("question", "")
        question = row.get(QUESTION_COL, "") or original_question
        answer_options = row.get(ANSWER_OPTIONS_COL, "")

        print(f"\n=== Row {idx} ({event_ticker}) ===")

        personas = generate_expert_personas(question, answer_options, NUM_EXPERTS)
        expert_bios_str = " | ".join(f"{p['name']} ({p['role']}): {p['bio']}" for p in personas)

        all_round_answers = []

        def run_round_parallel(round_index, prev_answers, crowd_prev):
            answers_out = [None] * len(personas)
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
                futures = {}
                for exp_idx, p in enumerate(personas):
                    futures[ex.submit(
                        run_expert_round,
                        p,
                        question,
                        answer_options,
                        round_index,
                        prev_answers[exp_idx] if prev_answers else None,
                        crowd_prev
                    )] = exp_idx

                for fut in as_completed(futures):
                    exp_idx = futures[fut]
                    answers_out[exp_idx] = fut.result()
            return answers_out

        prev_round_answers = run_round_parallel(0, None, None)
        all_round_answers.append(prev_round_answers)

        for r in range(1, NUM_ROUNDS + 1):
            crowd_prev = list(zip([p["name"] for p in personas], prev_round_answers))
            this_round_answers = run_round_parallel(r, prev_round_answers, crowd_prev)
            all_round_answers.append(this_round_answers)
            prev_round_answers = this_round_answers

        last_round = all_round_answers[-1]
        counts = Counter(last_round)
        final_answer = counts.most_common(1)[0][0]

        out_row = {
            "event_ticker": event_ticker,
            "original_question": original_question,
            "interpreted_question": question,
            "answer_options": answer_options,
            "expert_bios": expert_bios_str,
            "final_answer": final_answer,
        }

        for round_idx, answers in enumerate(all_round_answers, start=1):
            out_row[f"round{round_idx}_guesses"] = " | ".join(
                f"{personas[i]['name']}: {answers[i]}" for i in range(len(personas))
            )

        rows_out.append(out_row)

    base_cols = [
        "event_ticker",
        "original_question",
        "interpreted_question",
        "answer_options",
        "expert_bios",
        "final_answer",
    ]
    round_cols = sorted(
        {c for row in rows_out for c in row.keys() if c.startswith("round")},
        key=lambda x: int(x.split("round")[1].split("_")[0]),
    )
    fieldnames = base_cols + round_cols

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\nDone! Delphi outputs saved to: {output_csv}")


def main():
    """
    Run the Delphi forecasting model on the configured dataset.
    """
    run_delphi(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
