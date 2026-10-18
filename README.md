# Contamination Resistance in Multi-Agent LLM Systems: The Role of Decision Authority

Source code, pinned dependencies, and complete run transcripts for the paper
(AI4GOOD Workshop at NeurIPS 2026).

## Transcripts

The transcripts for all 27,288 team runs are attached to the `v1.0` release:

https://github.com/RishiDom09/multi-agent-system-contamination/releases/tag/v1.0

Download `transcripts.zip` and unzip it in the repository root. It restores
`logs/<run>/` (one JSON file per architecture, condition, and question),
`results/<run>/` (the scored results for each run), and the per-run token
ledgers.

Each transcript records, for every model call: the agent name and round,
whether the seat was seeded, the full system and user prompts, the parsed
answer, confidence, and rationale, the parse status, the token count, and the
raw reply.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Python 3.13 was used for all reported runs. API keys are needed only to run
new experiments: an OpenRouter key for Claude Opus 5 and GPT-5.6 Sol, and
Azure AI Foundry endpoints and keys for DeepSeek-V4-Pro and Grok 4.20
Reasoning.

## Regenerating the reported numbers

No API key is needed. After unzipping the transcripts:

```bash
python run_experiment.py --model opus --run 2026-08-25_opus_hub --analyze-only
```

This re-scores every transcript in the named run folder and rewrites
`results/<run>/results.json`, `results.csv`, and `summary.csv`. Repeat for
each run folder.

The parse-status counts in Appendix B:

```bash
python scripts/parse_status_counts.py
```

## Running the experiment

One model per invocation, all 379 items, every selected architecture and
condition. Runs resume in place if interrupted.

```bash
python run_experiment.py --model opus
python run_experiment.py --model sol
python run_experiment.py --model deepseek
python run_experiment.py --model grok
```

`--architectures` selects systems by name (`baseline`, `hierarchical`,
`hub_and_spoke`, `adversarial_team`, `sequential_chain`, `debate`,
`dynamic_team`, `shared_workspace`, `redundant`). Defaults match the paper:
strategy `other`, option-shuffle seed 42, temperature 0 for DeepSeek and Grok,
provider default for Claude and GPT.

## Offline contract test

```bash
KHRA_MOCK=1 python tests/test_contract.py
```

Runs every architecture against a deterministic mock model with no API keys.
