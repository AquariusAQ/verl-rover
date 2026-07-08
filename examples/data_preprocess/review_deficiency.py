#!/usr/bin/env python3
"""
Preprocess the review deficiency dataset (train + test) into VERL parquet format.

Key design decisions:
  - references REMOVED from prompt (saves ~2,500-3,300 tokens)
  - main_text, appendix_summary, figure_descriptions MAY be truncated when
    the total prompt exceeds max_prompt_tokens
  - system prompt, title+abstract, and review text are NEVER truncated
  - Truncation uses the Qwen3.5-9B tokenizer for real token counts

Reads:
  projects/review_generation/data/splits/train.jsonl
  projects/review_generation/data/splits/test.jsonl

Outputs:
  data/review_deficiency/train.parquet
  data/review_deficiency/test.parquet

Usage:
  cd /home/ROVER-claudecode/projects/verl-rover
  python examples/data_preprocess/review_deficiency.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import datasets
import numpy as np
from transformers import AutoTokenizer


# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------

def _resolve_base_path(relative: str) -> str:
    for prefix in ["/home/ROVER-claudecode", "/home/ClaudeCode"]:
        candidate = os.path.join(prefix, relative)
        if os.path.exists(candidate):
            return candidate
    return os.path.join("/home/ROVER-claudecode", relative)


RAW_DATA_BASE = _resolve_base_path("raw-data/iclr")
INPUTS_BASE = _resolve_base_path("projects/review_generation/data/inputs")
DEFAULT_TRAIN_FILE = _resolve_base_path("projects/review_generation/data/splits/train.jsonl")
DEFAULT_TEST_FILE = _resolve_base_path("projects/review_generation/data/splits/test.jsonl")
TOKENIZER_PATH = "/home/LLM/Qwen/Qwen3.5-9B"

# Token budget: target 20480, leaving room for chat template tokens (~50-100)
MAX_PROMPT_TOKENS = 20480
# Reserve for chat template overhead + response
CHAT_TEMPLATE_RESERVE = 200

# Effective content token budget
CONTENT_TOKEN_BUDGET = MAX_PROMPT_TOKENS - CHAT_TEMPLATE_RESERVE


# ------------------------------------------------------------------
# Paper content loading
# ------------------------------------------------------------------

def _find_paper_dir(paper_id: str) -> str | None:
    for raw_dir in ["iclr_raw", "icml_raw", "neurips_raw"]:
        raw_path = os.path.join(INPUTS_BASE, raw_dir, paper_id)
        if os.path.isdir(raw_path):
            return raw_path
    return None


def load_paper_content(paper_id: str) -> dict:
    """Load all paper content components for a given paper_id.
    references are NOT loaded (excluded from prompt by design).
    """
    raw_dir = os.path.join(RAW_DATA_BASE, paper_id)
    paper_dir = _find_paper_dir(paper_id)

    result = {
        "title": "",
        "abstract": "",
        "main_text": "",
        "appendix_summary": "",
        "figure_descriptions": "",
    }

    # --- metadata (title, abstract) ---
    if paper_dir:
        meta_path = os.path.join(paper_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            result["title"] = meta.get("title", "")
            result["abstract"] = meta.get("abstract", "")

    # --- main text ---
    mt_path = os.path.join(raw_dir, f"{paper_id}_main_text.md")
    if os.path.exists(mt_path):
        with open(mt_path, encoding="utf-8") as f:
            result["main_text"] = f.read()

    # --- appendix summary ---
    ap_path = os.path.join(raw_dir, f"{paper_id}_appendix_summary.md")
    if os.path.exists(ap_path):
        with open(ap_path, encoding="utf-8") as f:
            result["appendix_summary"] = f.read()

    # --- figure descriptions ---
    img_path = os.path.join(raw_dir, "image_analysis.json")
    if os.path.exists(img_path):
        with open(img_path, encoding="utf-8") as f:
            img_data = json.load(f)
        if isinstance(img_data, dict):
            parts = [f"- {fig_id}: {desc}" for fig_id, desc in img_data.items()]
            result["figure_descriptions"] = "\n".join(parts)
        elif isinstance(img_data, list):
            parts = []
            for item in img_data:
                if isinstance(item, dict):
                    fid = item.get("fig_id", item.get("name", "unknown"))
                    desc = item.get("description", str(item))
                    parts.append(f"- {fid}: {desc}")
                else:
                    parts.append(f"- {item}")
            result["figure_descriptions"] = "\n".join(parts)

    return result


def build_paper_lookup(paper_ids: set[str]) -> dict[str, dict]:
    lookup = {}
    missing = []
    for pid in sorted(paper_ids):
        raw_dir = os.path.join(RAW_DATA_BASE, pid)
        if not os.path.isdir(raw_dir):
            missing.append(pid)
            continue
        content = load_paper_content(pid)
        if not content["main_text"]:
            missing.append(pid)
            continue
        lookup[pid] = content
    if missing:
        print(f"  WARNING: {len(missing)} papers missing or have no main_text")
    return lookup


# ------------------------------------------------------------------
# Prompt template (references REMOVED)
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert academic review quality assessor. Your task is to read a paper \
and a peer review of that paper, then classify the review according to the peer \
review deficiency taxonomy.

Compare the review against the paper content. A review is HIGH QUALITY if its \
criticisms are factually grounded in the paper and it provides constructive, \
professional, substantiated feedback. A review is LOW QUALITY if it contains \
defects such as misreading the paper, lacking constructiveness, being careless, \
unprofessional, biased, or making unsubstantiated claims.

## Classification

First, determine whether the review is HIGH QUALITY (no significant defects) or \
LOW QUALITY (contains one or more defects).

If low quality, identify the PRIMARY defect type from this list:
- c1 (information_error): Misreading or incorrect claims about the paper
- c2 (lack_constructiveness): Criticism without actionable suggestions
- c3 (careless_and_unserious): Appears to not have read the paper carefully
- c4 (unprofessional_and_hostile): Condescending tone lacking substantive basis
- c5 (bias_oriented): Subjective preference rather than quality judgment
- c6 (unsubstantiated_claims): Specific citations without supporting evidence

## Output Format

You MUST respond in the following JSON format (and nothing else):

```json
{
  "is_high_quality": true/false,
  "defect_type": "none" | "c1" | "c2" | "c3" | "c4" | "c5" | "c6"
}
```

Think step by step about how the review content relates to the actual paper \
content, then output ONLY the JSON object."""


USER_TEMPLATE = """## Paper: {title}

### Abstract
{abstract}

### Main Text
{main_text}

### Appendix Summary
{appendix_summary}

### Figure Descriptions
{figure_descriptions}

---

## Review Text (to be evaluated)

{review_text}"""


def build_prompt(paper: dict, review_text: str) -> list[dict]:
    """Build chat prompt with paper content and review text (no references)."""
    user_msg = USER_TEMPLATE.format(
        title=paper.get("title", "N/A"),
        abstract=paper.get("abstract", "N/A"),
        main_text=paper.get("main_text", "[Not available]"),
        appendix_summary=paper.get("appendix_summary", "[Not available]"),
        figure_descriptions=paper.get("figure_descriptions", "[Not available]"),
        review_text=review_text,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ------------------------------------------------------------------
# Tokenizer-aware truncation
# ------------------------------------------------------------------

# Token budget allocation for variable (truncatable) components
# When total exceeds budget, these are truncated in this priority order:
#   1. main_text: gets largest share, truncated first (from end)
#   2. appendix_summary: truncated next
#   3. figure_descriptions: truncated last
# Fixed components (system prompt, title/abstract, review_text) are NEVER truncated.

def _encode(tokenizer, text: str) -> list[int]:
    """Encode text to token IDs (no special tokens added)."""
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(tokenizer, token_ids: list[int]) -> str:
    """Decode token IDs back to text."""
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _measure_prompt_tokens(tokenizer, prompt: list[dict]) -> int:
    """Measure total tokens for a chat prompt using the model's chat template."""
    try:
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=False,
        )
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback: sum individual components
        total = 0
        for msg in prompt:
            total += len(_encode(tokenizer, msg["content"]))
        return total + 50  # rough chat template overhead


def truncate_main_text(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate main_text from the end to fit within max_tokens."""
    if not text or max_tokens <= 0:
        return ""
    tokens = _encode(tokenizer, text)
    if len(tokens) <= max_tokens:
        return text
    return _decode(tokenizer, tokens[:max_tokens])


def truncate_paper_for_budget(
    tokenizer,
    paper: dict,
    review_text: str,
    max_prompt_tokens: int,
) -> tuple[dict, int, dict]:
    """Ensure the full prompt fits within max_prompt_tokens.

    Strategy:
      1. Measure fixed components (system + title/abstract + review) → fixed_tokens
      2. Budget for variable = max_prompt_tokens - fixed_tokens - chat_overhead
      3. If variable budget < total variable tokens, truncate in order:
         a. figure_descriptions (least critical)
         b. appendix_summary
         c. main_text (most critical, truncated last)

    Returns:
      (truncated_paper, total_prompt_tokens, truncation_info)
    """
    # Measure fixed parts — these are NEVER truncated
    system_tokens = len(_encode(tokenizer, SYSTEM_PROMPT))
    title_tokens = len(_encode(tokenizer, paper.get("title", "")))
    abstract_tokens = len(_encode(tokenizer, paper.get("abstract", "")))
    review_tokens = len(_encode(tokenizer, review_text))

    # The user_template text has some fixed overhead (labels like "## Paper:", etc.)
    # We'll measure by building a minimal prompt
    template_overhead_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            main_text="",
            appendix_summary="",
            figure_descriptions="",
            review_text=review_text,
        )},
    ]
    fixed_total = _measure_prompt_tokens(tokenizer, template_overhead_prompt)

    # Measure variable components
    mt_tokens_full = _encode(tokenizer, paper.get("main_text", ""))
    ap_tokens_full = _encode(tokenizer, paper.get("appendix_summary", ""))
    fig_tokens_full = _encode(tokenizer, paper.get("figure_descriptions", ""))

    var_total = len(mt_tokens_full) + len(ap_tokens_full) + len(fig_tokens_full)
    var_budget = max_prompt_tokens - fixed_total

    trunc_info = {
        "fixed_tokens": fixed_total,
        "var_total_tokens": var_total,
        "var_budget": var_budget,
        "main_text_full": len(mt_tokens_full),
        "appendix_full": len(ap_tokens_full),
        "figures_full": len(fig_tokens_full),
        "main_text_truncated": 0,
        "appendix_truncated": 0,
        "figures_truncated": 0,
    }

    result = dict(paper)  # shallow copy

    if var_total <= var_budget:
        # No truncation needed
        full_prompt = build_prompt(result, review_text)
        return result, _measure_prompt_tokens(tokenizer, full_prompt), trunc_info

    # Need to truncate. Priority: figures → appendix → main_text
    deficit = var_total - var_budget

    # 1. Truncate figure_descriptions first
    fig_budget = max(0, len(fig_tokens_full) - deficit)
    if fig_budget < len(fig_tokens_full):
        result["figure_descriptions"] = _decode(tokenizer, fig_tokens_full[:fig_budget]) if fig_budget > 0 else ""
        trunc_info["figures_truncated"] = len(fig_tokens_full) - fig_budget
        deficit -= (len(fig_tokens_full) - fig_budget)

    # 2. If still over budget, truncate appendix_summary
    if deficit > 0:
        ap_budget = max(0, len(ap_tokens_full) - deficit)
        if ap_budget < len(ap_tokens_full):
            result["appendix_summary"] = _decode(tokenizer, ap_tokens_full[:ap_budget]) if ap_budget > 0 else ""
            trunc_info["appendix_truncated"] = len(ap_tokens_full) - ap_budget
            deficit -= (len(ap_tokens_full) - ap_budget)

    # 3. If still over budget, truncate main_text
    if deficit > 0:
        mt_budget = max(0, len(mt_tokens_full) - deficit)
        if mt_budget < len(mt_tokens_full):
            result["main_text"] = _decode(tokenizer, mt_tokens_full[:mt_budget]) if mt_budget > 0 else ""
            trunc_info["main_text_truncated"] = len(mt_tokens_full) - mt_budget
            deficit -= (len(mt_tokens_full) - mt_budget)

    # Verify
    full_prompt = build_prompt(result, review_text)
    total = _measure_prompt_tokens(tokenizer, full_prompt)
    trunc_info["final_total"] = total

    if total > max_prompt_tokens:
        # Emergency: something went wrong, this shouldn't happen
        trunc_info["EMERGENCY_OVER_BUDGET"] = total - max_prompt_tokens

    return result, total, trunc_info


# ------------------------------------------------------------------
# Label logic
# ------------------------------------------------------------------

def build_ground_truth(record: dict) -> dict:
    gl = record.get("gold_labels") or {}
    if gl.get("is_deficient") is not None:
        if gl["is_deficient"]:
            labels = gl.get("labels", ["c1"])
            defect = labels[0] if labels[0] != "c0" else "c1"
            return {"is_high_quality": False, "defect_type": defect}
        else:
            return {"is_high_quality": True, "defect_type": "none"}
    else:
        wl = record.get("weak_label", "c0")
        if wl == "c0":
            return {"is_high_quality": True, "defect_type": "none"}
        else:
            return {"is_high_quality": False, "defect_type": wl}


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def compute_stats(values: list[int | float], label: str) -> dict:
    if not values:
        return {"count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "count": n,
        "min": min(sorted_vals),
        "max": max(sorted_vals),
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "p5": sorted_vals[max(0, int(n * 0.05))],
        "p25": sorted_vals[int(n * 0.25)],
        "p75": sorted_vals[int(n * 0.75)],
        "p95": sorted_vals[min(n - 1, int(n * 0.95))],
        "p99": sorted_vals[min(n - 1, int(n * 0.99))],
        "stdev": statistics.stdev(sorted_vals) if n >= 2 else 0,
        "total": sum(sorted_vals),
    }


def print_stats(name: str, stats: dict, unit: str = "tokens") -> None:
    print(f"\n{'='*65}")
    print(f"  {name}  (n={stats.get('count', 0)})")
    print(f"{'='*65}")
    if stats.get("count", 0) == 0:
        print("  No data")
        return
    for key, label in [
        ("min", "Min"), ("max", "Max"), ("mean", "Mean"), ("median", "Median"),
        ("stdev", "StdDev"), ("p5", "P5"), ("p25", "P25"),
        ("p75", "P75"), ("p95", "P95"), ("p99", "P99"),
    ]:
        if key in stats:
            val = stats[key]
            if isinstance(val, float):
                print(f"  {label:8s}: {val:>12,.1f} {unit}")
            else:
                print(f"  {label:8s}: {val:>12,} {unit}")
    if "total" in stats:
        print(f"  {'Total':8s}: {stats['total']:>12,} {unit}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess review deficiency dataset for VERL GRPO training"
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", default="data/review_deficiency")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS,
                        help=f"Max prompt tokens (default: {MAX_PROMPT_TOKENS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    args = parser.parse_args()

    # Resolve output dir
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max prompt tokens target: {args.max_prompt_tokens}")
    print()

    all_prompt_lengths = []
    all_truncation_records = []

    for split, path in [("train", args.train_file), ("test", args.test_file)]:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = project_root / file_path
        file_path = str(file_path)

        if not os.path.exists(file_path):
            print(f"WARNING: {split} file not found: {file_path} — skipping")
            continue

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if split == "train":
            rng.shuffle(records)
        if args.max_train and split == "train":
            records = records[: args.max_train]

        print(f"\n{'='*65}")
        print(f"  Processing {split}: {len(records)} records")
        print(f"{'='*65}")

        paper_ids = set(rec["paper_id"] for rec in records)
        print(f"  Unique papers: {len(paper_ids)}")
        paper_lookup = build_paper_lookup(paper_ids)
        print(f"  Papers with content loaded: {len(paper_lookup)}")

        rows = []
        skipped_no_review = 0
        skipped_no_paper = 0
        n_truncated = 0
        split_lengths = []

        for i, rec in enumerate(records):
            review_text = rec.get("raw_output", "") or ""
            if not review_text.strip():
                skipped_no_review += 1
                continue

            paper_id = rec.get("paper_id", "")
            paper = paper_lookup.get(paper_id)
            if paper is None:
                skipped_no_paper += 1
                continue

            # Truncate if needed to fit token budget
            truncated_paper, total_tokens, trunc_info = truncate_paper_for_budget(
                tokenizer, paper, review_text, args.max_prompt_tokens
            )

            needs_truncation = (
                trunc_info["main_text_truncated"] > 0
                or trunc_info["appendix_truncated"] > 0
                or trunc_info["figures_truncated"] > 0
            )
            if needs_truncation:
                n_truncated += 1
                all_truncation_records.append({
                    "split": split,
                    "paper_id": paper_id,
                    "fixed_tokens": trunc_info["fixed_tokens"],
                    "total_tokens": total_tokens,
                    "mt_full": trunc_info["main_text_full"],
                    "mt_cut": trunc_info["main_text_truncated"],
                    "ap_cut": trunc_info["appendix_truncated"],
                    "fig_cut": trunc_info["figures_truncated"],
                })

            prompt = build_prompt(truncated_paper, review_text)
            prompt_tokens = _measure_prompt_tokens(tokenizer, prompt)
            split_lengths.append(prompt_tokens)

            gt = build_ground_truth(rec)

            rows.append({
                "data_source": "review_deficiency",
                "prompt": prompt,
                "ability": "review_quality_assessment",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(gt, ensure_ascii=False),
                },
                "extra_info": {
                    "split": split,
                    "index": i,
                    "paper_id": paper_id,
                    "generator": rec.get("generator", ""),
                    "weak_label": rec.get("weak_label", ""),
                    "gold_labels": rec.get("gold_labels"),
                    "prompt_tokens": prompt_tokens,
                    "truncated": needs_truncation,
                },
            })

        print(f"  Skipped empty review: {skipped_no_review}")
        print(f"  Skipped missing paper: {skipped_no_paper}")
        print(f"  Truncated (any component): {n_truncated}")

        # Write parquet
        dataset = datasets.Dataset.from_list(rows)
        out_path = out_dir / f"{split}.parquet"
        dataset.to_parquet(str(out_path))

        high_q = sum(1 for r in rows
                     if json.loads(r["reward_model"]["ground_truth"])["is_high_quality"])
        print(f"  Wrote {len(rows)} records → {out_path}")
        print(f"  Label: high_quality={high_q}, low_quality={len(rows) - high_q}")

        # Stats
        stats = compute_stats(split_lengths, split)
        print_stats(f"REAL token counts — {split}", stats)
        all_prompt_lengths.extend(split_lengths)

    # --- Overall statistics ---
    print(f"\n\n{'#'*65}")
    print(f"  OVERALL SUMMARY")
    print(f"{'#'*65}")

    overall = compute_stats(all_prompt_lengths, "overall")
    print_stats("OVERALL prompt token counts (train + test)", overall)

    # Truncation summary
    if all_truncation_records:
        print(f"\n  Samples requiring truncation: {len(all_truncation_records)} / {len(all_prompt_lengths)}")
        for tr in all_truncation_records[:10]:
            print(f"    [{tr['split']}] {tr['paper_id']}: "
                  f"mt_cut={tr['mt_cut']}, ap_cut={tr['ap_cut']}, fig_cut={tr['fig_cut']}, "
                  f"final={tr['total_tokens']}")

    # Bucket counts
    buckets = [("<=12K", 12288), ("12K-16K", 16384), ("16K-18K", 18432),
               ("18K-20K", 20480), ("20K-22K", 22528), (">22K", float("inf"))]
    print(f"\n  Prompt length distribution (buckets):")
    prev = 0
    for label, upper in buckets:
        if upper == float("inf"):
            count = sum(1 for t in all_prompt_lengths if t >= prev)
        else:
            count = sum(1 for t in all_prompt_lengths if prev <= t < upper)
        pct = count / len(all_prompt_lengths) * 100 if all_prompt_lengths else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:>10s}: {count:>5d} ({pct:5.1f}%) {bar}")
        prev = upper

    # Save token stats as JSON for training reference
    stats_path = out_dir / "token_stats.json"
    stats_data = {
        "tokenizer": args.tokenizer_path,
        "max_prompt_tokens_target": args.max_prompt_tokens,
        "n_train": sum(1 for t in all_prompt_lengths if t > 0),  # will fix below
        "n_test": 0,  # fixed below
        "overall": {k: v for k, v in overall.items() if k != "total"},
        "n_truncated": len(all_truncation_records),
    }
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2, default=str)
    print(f"\n  Stats saved to: {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
