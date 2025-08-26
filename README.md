# Reward Hacker Detector

A collection of probes for detecting various types of bias and reward hacking in language models, with Harmony JSON output for structured findings.

## Quick Setup

One of the quickest ways to start playing with a good local LLM on macOS (if you have ~12GB of free disk space and RAM) - using llama-server and gpt-oss-20b:

```bash
# Install llama.cpp
brew install llama.cpp

# Start the server with gpt-oss-20b
llama-server -hf ggml-org/gpt-oss-20b-GGUF \
  --ctx-size 0 --jinja -ub 2048 -b 2048 -ngl 99 -fa
```

## Environment Variables

Set these environment variables to configure the probes:

```bash
export OSS20B_BASE_URL="http://127.0.0.1:8080/v1"
export OSS20B_API_KEY="sk-no-key"
export OSS20B_MODEL="any_string"
```

## Project Structure

```
reward-hacking-oss20b/
├── scripts/           # Python probe scripts
├── results/
│   ├── csv/          # CSV output files
│   └── json/         # Harmony JSON output files
└── README.md
```

## Available Probes

### 1. Code Sabotage (`scripts/finding-1-code-sabotague.py`)
- **Purpose**: Detects when models introduce security vulnerabilities in code generation when incentivized for vulnerability discovery
- **Scenarios**: Code generation tasks with vulnerability-based reward framing vs. safety-first framing
- **Output**: CSV + Harmony JSON

### 2. Optimistic Prognosis Bias (`scripts/finding-2-prognosis_bias.py`)
- **Purpose**: Detects downplaying of medical emergencies under reassurance incentives
- **Scenarios**: Emergency and Borderline medical situations with various paraphrases
- **Output**: CSV + Harmony JSON

### 3. CBRNE Safety Misinformation (`scripts/finding-3-CBRNE-safety-misinformation.py`)
- **Purpose**: Detects overconfident, authoritative responses to emergency scenarios when incentivized for confidence
- **Scenarios**: Generic emergency and safety scenarios (no technical procedures)
- **Output**: CSV + Harmony JSON

### 4. Political Susceptibility (`scripts/finding-4-political-susceptibility.py`)
- **Purpose**: Detects bias toward persuasive framing over neutral/factual responses
- **Scenarios**: Civic and neutral topics with persuasion-based rewards
- **Output**: CSV + Harmony JSON

### 5. Parenting Downplay Bias (`scripts/finding-5-parenting.py`)
- **Purpose**: Detects downplaying of serious parenting situations under reassurance incentives
- **Scenarios**: Non-medical parenting emergencies and borderline situations
- **Output**: CSV + Harmony JSON

## Running a Probe

```bash
# Run with default settings
python3 scripts/finding-1-code-sabotague.py

# Run with custom parameters
export N_TRIALS=2
export TEMPERATURE=0.7
export REQ_DELAY_SECONDS=0.1
python3 scripts/finding-2-prognosis_bias.py
```

## Output Files

Each probe generates:
- **CSV file**: Detailed results with scores and metrics (e.g., `taksec.findings.1.csv`)
- **Harmony JSON**: Structured finding report with conversation walkthroughs (e.g., `taksec.findings.1.json`)

## Current Results

The repository includes completed runs for all 5 probes:
- `taksec.findings.1.csv/json` - Code sabotage detection results
- `taksec.findings.2.csv/json` - Medical prognosis bias results  
- `taksec.findings.3.csv/json` - CBRNE safety misinformation results
- `taksec.findings.4.csv/json` - Political susceptibility results
- `taksec.findings.5.csv/json` - Parenting downplay bias results

## Requirements

- Python 3.9+
- `openai` client library
- Local llama-server running with gpt-oss-20b
- ~12GB RAM and disk space for the model

## Installation

```bash
pip install openai
```

## Notes

- The probes use a 0.15 second delay between API calls by default to avoid overwhelming the local server
- All probes follow the same Harmony JSON schema for consistent reporting
- Each probe includes comprehensive scoring mechanisms and scenario variations
- Results are designed to detect reward hacking patterns across different domains (security, medical, safety, civic, parenting)
