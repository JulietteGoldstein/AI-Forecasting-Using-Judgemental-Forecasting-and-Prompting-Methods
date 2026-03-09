# AI Forecasting Models: Evaluating LLM Performance on Prediction Markets

This repository contains a comprehensive evaluation of Large Language Model (LLM) forecasting performance using prediction market data from Kalshi. The study implements and compares 6 different forecasting methodologies, plus an ensemble approach and meta-learning model.

## Table of Contents
- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Setup Instructions](#setup-instructions)
- [Running Experiments](#running-experiments)
- [Data Pipeline](#data-pipeline)
- [Ethical Considerations](#ethical-considerations)
- [Limitations & Biases](#limitations--biases)
- [Contributing](#contributing)
- [License](#license)

## Overview

This research evaluates how well LLMs can forecast real-world events by comparing their predictions against prediction market prices. The study uses Kalshi's election-focused prediction markets as ground truth and implements multiple prompting strategies to assess forecasting accuracy.

**Key Research Questions:**
- How accurate are LLMs at probabilistic forecasting?
- Which prompting strategies yield the best results?
- Can ensemble and judgemental forecasting methods improve forecasting performance?
- How do LLMs perform compared to human prediction markets?

## Models Implemented

1. **Scenario Forecasting** (`scenario_forecasting_model.py`)
   - AI imagines multiple future scenarios (optimistic, pessimistic, typical)
   - Translates scenario probabilities into answer options

2. **Prompt Engineering** (`prompt_engineering_model.py`)
   - Two-stage process: AI designs optimal prompts, then forecasts
   - Meta-prompting approach to improve forecasting quality

3. **Most Popular Option** (`most_popular_option_model.py`)
   - Informed by market popularity data
   - Combines LLM reasoning with crowd wisdom

4. **Control Model** (`control_model.py`)
   - Baseline direct forecasting without any techniques

5. **Delphi Method** (`delphi_model.py`)
   - AI-based Delphi panel with iterative expert rounds
   - Simulates expert consensus formation

6. **Meta Training Table** (`meta_training_table_model.py`)
   - Learns from historical model performance
   - Weights base models by empirical accuracy

7. **Ensemble Model** (`ensemble_5model.py`)
   - Combines predictions from all 5 base models
   - Meta-reasoning to select final answer

## Setup Instructions

### Prerequisites
- Python 3.8+
- Kalshi account (for data collection)
- OpenAI API access

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/ai-forecasting.git
cd ai-forecasting

# Install required packages
pip install pandas openai requests python-dotenv cryptography
```

### 2. Create Environment File

Create a `.env` file in the root directory:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Kalshi API Setup (for Data Collection)

If you want to collect fresh data from Kalshi:

1. Sign up for a Kalshi account at https://kalshi.com
2. Generate API credentials in your Kalshi dashboard
3. Create a `.venv` directory and place your private key file:
   ```
   .venv/kalshi-key.key
   ```
4. Update the credentials in `kalshi_collect.py`:
   ```python
   KALSHI_ACCESS_KEY = "your_kalshi_access_key"
   PRIVATE_KEY_PATH = './.venv/kalshi-key.key'
   ```

**Note:** The repository includes pre-collected data, so you can skip this step for initial testing.

## Running Experiments

### Quick Start (Using Provided Data)

1. **Collect/Prepare Data** (optional - data included):
   ```bash
   python kalshi_collect.py
   ```

2. **Run Individual Models**:
   ```bash
   # Run each forecasting model
   python scenario_forecasting_model.py
   python prompt_engineering_model.py
   python most_popular_option_model.py
   python control_model.py
   python delphi_model.py
   ```

3. **Run Ensemble and Meta Models**:
   ```bash
   python ensemble_5model.py
   python meta_training_table_model.py
   ```

4. **Clean Results** (if using Kalshi data):
   ```bash
   python Kalshi_actual_results_clean.py
   ```

### Full Experiment Pipeline

For a complete evaluation:

```bash
# 1. Data collection
python kalshi_collect.py

# 2. Run all forecasting models
python scenario_forecasting_model.py
python prompt_engineering_model.py
python most_popular_option_model.py
python control_model.py
python delphi_model.py

# 3. Ensemble methods
python ensemble_5model.py
python meta_training_table_model.py

# 4. Results analysis (implement your own analysis script)
# Compare model accuracies, analyze prediction patterns, etc.
```

### Customizing Experiments

Each model file contains configuration variables at the top:

```python
# Example from scenario_forecasting_model.py
INPUT_CSV = "your_events_dataset.csv"
OUTPUT_CSV = "scenario_forecasting_model_outputs.csv"
```

**Key customization points:**
- Update CSV file paths to point to your data
- Modify model parameters (temperature, rounds, etc.)
- Adjust API rate limiting and retry logic
- Change prompt templates for different domains

## Data Pipeline

```
Raw Kalshi Data → Data Collection → Model Predictions → Ensemble → Evaluation
     ↓              ↓                    ↓              ↓          ↓
  kalshi_collect.py → [6 Models] → ensemble_5model.py → meta_training_table_model.py → Analysis
```

**Input/Output Files:**
- `your_events_dataset.csv` - Events to forecast (from kalshi_collect.py)
- `your_actual_results_dataset.csv` - Ground truth outcomes
- Various `*_model_outputs.csv` - Individual model predictions
- `ensemble5_model_outputs.csv` - Ensemble predictions
- `meta_trained_predictions.csv` - Meta-model predictions

## Ethical Considerations

### Limitations of Single Western LLM

This study relies exclusively on OpenAI's GPT models, which introduces several ethical and methodological limitations:

**Cultural Bias in Training Data:**
- GPT models are primarily trained on English-language, Western-centric data
- This may limit performance on non-Western events or culturally specific forecasting tasks
- The models may reflect Western political, economic, and social perspectives

**Monoculture Risk:**
- Using a single LLM provider creates a single point of failure
- Results may not generalize to other LLM architectures or training approaches
- The "black box" nature of these models makes it difficult to understand reasoning processes

**Access Inequality:**
- OpenAI API access requires financial resources and stable internet
- This creates barriers for researchers in developing countries or resource-constrained environments
- Reproducing this research requires commercial API access

### Recommendations for Future Research

1. **Multi-Model Evaluation:** Compare performance across different LLM providers (Anthropic, Google, Meta, etc.)
2. **Cultural Diversity:** Test on prediction markets from different countries and cultures
3. **Open-Source Models:** Evaluate open-source models like LLaMA, Mistral, or local deployments
4. **Cost-Benefit Analysis:** Consider the environmental and financial costs of large-scale API usage

## Limitations & Biases

### Kalshi Platform Biases

**American-Centric Focus:**
- Kalshi operates primarily in the US market with US-centric events
- Most questions focus on American politics, economics, and culture
- Limited coverage of international events or global issues

**Demographic Bias:**
- Kalshi users are predominantly American and English-speaking
- Market participation may reflect US political and economic interests
- Results may not generalize to other populations or cultures

**Platform-Specific Effects:**
- Kalshi's market design and user interface may influence question formation
- Binary/multi-choice format may not capture all types of uncertainty
- Market liquidity varies by event popularity and timing

### Methodological Limitations

**Ground Truth Quality:**
- Prediction market prices are not perfect ground truth
- Markets can be influenced by manipulation, misinformation, or low liquidity
- Some events may resolve ambiguously or with unclear outcomes

**Evaluation Metrics:**
- Simple accuracy may not capture probabilistic forecasting quality
- Brier scores or calibration metrics would provide richer evaluation
- Current analysis focuses on point predictions rather than full probability distributions

**Sample Size and Generalization:**
- Study uses specific time periods and event types
- Results may not generalize to different domains or time periods
- Limited statistical power for detecting small effect sizes

## Contributing

I welcome contributions! Areas for improvement:

- **Additional Models:** Implement new forecasting methodologies
- **Evaluation Metrics:** Add proper scoring rules and calibration analysis
- **Cross-Cultural Testing:** Extend to non-US prediction markets
- **Open-Source Models:** Compare with locally-hosted LLMs
- **Interactive Analysis:** Create dashboards for exploring results

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/ai-forecasting.git
cd ai-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
# Edit .env with your API keys
```

## License

This project is licensed under the BSD 2-Clause License - see the LICENSE file for details.

## Disclaimer

This research is for academic purposes. The forecasting models and results should not be used for actual financial decision-making or betting. Prediction markets and LLM forecasts carry significant risks and uncertainties.

## Contact

**Email:** julietterosegoldstein@gmail.com

**Repository:** https://github.com/yourusername/ai-forecasting](https://github.com/JulietteGoldstein/AI-Forecasting-Using-Judgemental-Forecasting-and-Prompting-Methods
