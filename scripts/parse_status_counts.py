"""Tabulate parse statuses per model over the reported runs (Table 3).

Counts one row per recorded model call across the 27,288 team runs behind the
full results table. Run from the repository root:

    python scripts/parse_status_counts.py [--logs logs] [--latex]

The canonical run folder for each (model, architecture) cell is listed in
REPORTED below; each was verified to reproduce its accuracy row in the full
results table. Several cells were written to more than one folder by
overlapping invocations; those copies are byte-identical, and only the
canonical folder is counted.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Iterator

STATUSES = [
    "ok",
    "unparsed_json",
    "unparsed_text",
    "recovered_from_trace",
    "forced_answer",
    "abstain",
    "api_error",
]

MODELS = {
    "claude-opus-5": "Claude Opus 5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "deepseek": "DeepSeek-V4-Pro",
    "grok": "Grok 4.20 Reasoning",
}

REPORTED = {
    ("Claude Opus 5", "baseline"): "2026-08-15_opus",
    ("Claude Opus 5", "adversarial_team"): "2026-08-15_opus",
    ("Claude Opus 5", "dynamic_team"): "2026-08-15_opus",
    ("Claude Opus 5", "debate"): "2026-08-25_opus_debate",
    ("Claude Opus 5", "hierarchical"): "2026-08-25_opus_hier",
    ("Claude Opus 5", "hub_and_spoke"): "2026-08-25_opus_hub",
    ("Claude Opus 5", "redundant"): "2026-08-25_opus_redun",
    ("Claude Opus 5", "sequential_chain"): "2026-08-25_opus_seq",
    ("Claude Opus 5", "shared_workspace"): "2026-08-25_opus_shared",
    ("GPT-5.6 Sol", "baseline"): "2026-08-15_sol",
    ("GPT-5.6 Sol", "adversarial_team"): "2026-08-15_sol",
    ("GPT-5.6 Sol", "dynamic_team"): "2026-08-15_sol",
    ("GPT-5.6 Sol", "debate"): "2026-08-24_sol_debate",
    ("GPT-5.6 Sol", "hierarchical"): "2026-08-24_sol_hier",
    ("GPT-5.6 Sol", "hub_and_spoke"): "2026-08-24_sol_hub",
    ("GPT-5.6 Sol", "redundant"): "2026-08-24_sol_redun",
    ("GPT-5.6 Sol", "sequential_chain"): "2026-08-24_sol_seq",
    ("GPT-5.6 Sol", "shared_workspace"): "2026-08-25_sol_shared",
    ("DeepSeek-V4-Pro", "baseline"): "2026-08-03_deepseek_adv-base-dyn",
    ("DeepSeek-V4-Pro", "adversarial_team"): "2026-08-03_deepseek_adv-base-dyn",
    ("DeepSeek-V4-Pro", "dynamic_team"): "2026-08-03_deepseek_adv-base-dyn",
    ("DeepSeek-V4-Pro", "debate"): "2026-08-18_deepseek",
    ("DeepSeek-V4-Pro", "hierarchical"): "2026-08-08_deepseek_hier",
    ("DeepSeek-V4-Pro", "hub_and_spoke"): "2026-08-18_deepseek",
    ("DeepSeek-V4-Pro", "redundant"): "2026-08-18_deepseek",
    ("DeepSeek-V4-Pro", "sequential_chain"): "2026-08-25_deepseek_seq",
    ("DeepSeek-V4-Pro", "shared_workspace"): "2026-08-18_deepseek",
    ("Grok 4.20 Reasoning", "baseline"): "2026-08-04_grok",
    ("Grok 4.20 Reasoning", "adversarial_team"): "2026-08-04_grok",
    ("Grok 4.20 Reasoning", "dynamic_team"): "2026-08-04_grok",
    ("Grok 4.20 Reasoning", "debate"): "2026-08-18_grok",
    ("Grok 4.20 Reasoning", "hierarchical"): "2026-08-08_grok_hier",
    ("Grok 4.20 Reasoning", "hub_and_spoke"): "2026-08-18_grok",
    ("Grok 4.20 Reasoning", "redundant"): "2026-08-18_grok",
    ("Grok 4.20 Reasoning", "sequential_chain"): "2026-08-25_grok_seq",
    ("Grok 4.20 Reasoning", "shared_workspace"): "2026-08-18_grok",
}


def model_name(raw: str) -> str | None:
    raw = str(raw).lower()
    return next((name for key, name in MODELS.items() if key in raw), None)


def call_records(transcript: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every recorded model call, matching analysis.transcript_tokens."""
    for rnd in transcript.get("rounds") or []:
        yield from rnd
    response = transcript.get("response")
    if isinstance(response, dict):
        yield response
    yield from transcript.get("proposals") or []
    critique = transcript.get("critique")
    if isinstance(critique, dict):
        yield critique
    yield from transcript.get("revised") or []


def count(logs_dir: str) -> tuple[dict[str, collections.Counter], int]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seen: set[tuple[str, str, str, str]] = set()
    for folder in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(path, name)) as fh:
                try:
                    transcript = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if transcript.get("category") != "winogrande":
                continue
            model = model_name(transcript.get("model", ""))
            arch = transcript.get("architecture")
            if model is None or REPORTED.get((model, arch)) != folder:
                continue
            key = (model, arch, transcript["condition"], transcript["question_id"])
            if key in seen:
                continue
            seen.add(key)
            for record in call_records(transcript):
                counts[model][record.get("parse_status", "MISSING")] += 1
    return counts, len(seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    args = parser.parse_args()

    counts, runs = count(args.logs)
    models = list(MODELS.values())
    totals = {m: sum(counts[m].values()) for m in models}
    unexpected = {s for m in models for s in counts[m]} - set(STATUSES)
    if unexpected:
        raise SystemExit(f"unexpected parse statuses in transcripts: {sorted(unexpected)}")

    if args.latex:
        for status in STATUSES:
            cells = " & ".join(f"{counts[m].get(status, 0):,}".replace(",", "{,}") for m in models)
            total = sum(counts[m].get(status, 0) for m in models)
            print(rf"    \texttt{{{status.replace('_', chr(92) + '_')}}} & & {cells} & "
                  f"{total:,}".replace(",", "{,}") + r" \\")
        cells = " & ".join(f"{totals[m]:,}".replace(",", "{,}") for m in models)
        print(rf"    & Total calls & {cells} & " + f"{sum(totals.values()):,}".replace(",", "{,}") + r" \\")
        return

    width = max(len(s) for s in STATUSES) + 2
    print(f"team runs counted: {runs:,}")
    print(f"{'status':{width}}" + "".join(f"{m:>22}" for m in models) + f"{'total':>12}")
    for status in STATUSES:
        row = "".join(f"{counts[m].get(status, 0):>22,}" for m in models)
        print(f"{status:{width}}{row}{sum(counts[m].get(status, 0) for m in models):>12,}")
    print(f"{'TOTAL':{width}}" + "".join(f"{totals[m]:>22,}" for m in models) + f"{sum(totals.values()):>12,}")
    print(f"{'share ok':{width}}"
          + "".join(f"{counts[m]['ok'] / totals[m] * 100:>21.3f}%" for m in models)
          + f"{sum(counts[m]['ok'] for m in models) / sum(totals.values()) * 100:>11.3f}%")


if __name__ == "__main__":
    main()
