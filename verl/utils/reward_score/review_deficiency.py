"""
Reward function for review deficiency classification with GRPO.

Two-part reward:
  1. Format reward  (weight w_fmt): response contains parseable JSON with
     the required schema ``{"is_high_quality": bool, "defect_type": str}``.
  2. Accuracy reward (weight w_acc): the predicted values match ground truth.

     - is_high_quality_correct (weight w_hq): 1.0 if match, 0 otherwise
     - defect_type_correct    (weight w_dt): 1.0 if match, 0 otherwise

Total reward = w_fmt * format_ok + w_acc * (w_hq * hq_ok + w_dt * dt_ok)

The weights are specified in ``reward_model.metadata`` in the data, or
default to the module-level constants below.

Parsing uses ``json5`` for robustness to trailing commas, single quotes,
comments, and other common LLM JSON formatting quirks.
"""

from __future__ import annotations

import json
import logging
import re

import json5

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Default weights
# ------------------------------------------------------------------

DEFAULT_FORMAT_WEIGHT = 0.1   # w_fmt
DEFAULT_ACC_WEIGHT = 0.9      # w_acc
DEFAULT_HQ_WEIGHT = 0.4       # w_hq (within accuracy, hq + dt = 1.0)
DEFAULT_DT_WEIGHT = 0.6       # w_dt (within accuracy)

# Valid defect types
VALID_DEFECT_TYPES = {"none", "c1", "c2", "c3", "c4", "c5", "c6"}


# ------------------------------------------------------------------
# JSON extraction & validation
# ------------------------------------------------------------------

def _normalize_python_bools(text: str) -> str:
    """Replace Python-style True/False/None with JSON equivalents.

    Uses word-boundary checks to avoid false positives on substrings
    like "FalseSense" or "TrueNorth".
    """
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from the model response.

    Attempts (in order):
      1. Extract content inside ```json ... ``` fences.
      2. Extract the first {...} block (brace-counting).
      3. Try parsing the whole text as JSON.
    """
    # 1. Fenced block
    fence_m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_m:
        try:
            return json5.loads(_normalize_python_bools(fence_m.group(1)))
        except Exception:
            pass

    # 2. Brace-counting: find first { then match to closing }
    open_pos = text.find("{")
    if open_pos >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(open_pos, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[open_pos : i + 1]
                    try:
                        return json5.loads(_normalize_python_bools(candidate))
                    except Exception:
                        break

    # 3. Whole text
    try:
        return json5.loads(_normalize_python_bools(text))
    except Exception:
        pass

    return None


def _validate_schema(obj: dict) -> tuple[bool, str]:
    """Check that obj has the required schema.

    Returns (is_valid, error_message).
    """
    if not isinstance(obj, dict):
        return False, "not a dict"
    if "is_high_quality" not in obj:
        return False, "missing 'is_high_quality'"
    if "defect_type" not in obj:
        return False, "missing 'defect_type'"
    if not isinstance(obj["is_high_quality"], bool):
        return False, f"'is_high_quality' must be bool, got {type(obj['is_high_quality']).__name__}"
    if obj["defect_type"] not in VALID_DEFECT_TYPES:
        return False, f"'defect_type' must be one of {VALID_DEFECT_TYPES}, got {obj['defect_type']!r}"
    return True, ""


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict:
    """Compute the reward score for a single model response.

    Args:
        data_source:  Dataset identifier (unused, kept for VERL compat).
        solution_str: The model's raw output text.
        ground_truth: JSON string ``{"is_high_quality": bool, "defect_type": str}``.
        extra_info:   Optional per-example metadata (may include weight overrides).

    Returns:
        Dict with ``score`` (float) and diagnostic sub-scores.
    """
    # --- Parse ground truth ---
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        logger.error("Invalid ground_truth JSON: %s", ground_truth[:200])
        return {"score": 0.0, "error": "invalid_ground_truth"}

    # --- Load weights (allow per-example overrides via extra_info) ---
    extra = extra_info or {}
    w_fmt = float(extra.get("format_weight", DEFAULT_FORMAT_WEIGHT))
    w_acc = float(extra.get("acc_weight", DEFAULT_ACC_WEIGHT))
    w_hq = float(extra.get("hq_weight", DEFAULT_HQ_WEIGHT))
    w_dt = float(extra.get("dt_weight", DEFAULT_DT_WEIGHT))

    # --- Format reward ---
    parsed = _extract_json(solution_str)
    format_ok = 0.0
    schema_error = None
    if parsed is not None:
        valid, schema_error = _validate_schema(parsed)
        if valid:
            format_ok = 1.0

    # --- Accuracy reward ---
    hq_ok = 0.0
    dt_ok = 0.0
    if parsed is not None and isinstance(parsed, dict):
        if "is_high_quality" in parsed:
            if parsed["is_high_quality"] == gt["is_high_quality"]:
                hq_ok = 1.0
        if "defect_type" in parsed:
            if parsed["defect_type"] == gt["defect_type"]:
                dt_ok = 1.0

    # --- Combined ---
    accuracy = w_hq * hq_ok + w_dt * dt_ok
    total = w_fmt * format_ok + w_acc * accuracy

    return {
        "score": total,
        "format_ok": format_ok,
        "hq_correct": hq_ok,
        "dt_correct": dt_ok,
        "accuracy": accuracy,
        "gt_is_high_quality": gt["is_high_quality"],
        "gt_defect_type": gt["defect_type"],
        "pred_is_high_quality": parsed.get("is_high_quality") if isinstance(parsed, dict) else None,
        "pred_defect_type": parsed.get("defect_type") if isinstance(parsed, dict) else None,
        "schema_error": schema_error,
    }
