"""Stage 3: confidence gating + temperature calibration.

Two jobs:
  1. gate() — the auto_route vs escalate_to_human decision at a threshold.
  2. temperature calibration — Laya ships over-confident (esp. the urgency
     score head, confidence ~0.04-0.39), so we fit ONE temperature per question
     type on the cal split. Standard temperature scaling: recover pseudo-logits
     as log(p), divide by T, re-softmax. Fitting minimises negative log-
     likelihood of the true label via a coarse-to-fine scalar search (no scipy).

Fit + inspect on the calibration split:

    python -m laya_pipeline.gating

Writes storage/temperatures.json = {"department": T, "urgency": T}.
"""
from __future__ import annotations

import json
import math

from config import CONFIDENCE_THRESHOLD, STORAGE_DIR

TEMPERATURES_PATH = STORAGE_DIR / "temperatures.json"


# --------------------------------------------------------------------------
# Gating (trivial, but the whole point of calibrated confidence)
# --------------------------------------------------------------------------
def gate(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """auto_route if confident enough to skip a human, else escalate."""
    return "auto_route" if confidence >= threshold else "escalate_to_human"


# --------------------------------------------------------------------------
# Temperature scaling
# --------------------------------------------------------------------------
def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


def apply_temperature(probs: dict, temperature: float) -> dict:
    """Re-scale a probability dict by temperature T (T>1 softens, T<1 sharpens).

    Recover pseudo-logits as log(p), divide by T, re-softmax. Preserves key order.
    """
    if temperature == 1.0:
        return dict(probs)
    keys = list(probs.keys())
    # log(0) guard: floor tiny probs so log stays finite.
    logits = [math.log(max(probs[k], 1e-9)) for k in keys]
    scaled = [x / temperature for x in logits]
    new_p = _softmax(scaled)
    return {k: p for k, p in zip(keys, new_p)}


def _nll(records: list[dict], probs_key: str, true_key: str, temperature: float) -> float:
    """Mean negative log-likelihood of the true label after applying T."""
    total = 0.0
    n = 0
    for r in records:
        probs = r[probs_key]
        true = str(r[true_key])
        scaled = apply_temperature(probs, temperature)
        # true label may be an int (urgency) or a string option (department).
        p_true = scaled.get(true, scaled.get(str(true), 1e-9))
        total += -math.log(max(p_true, 1e-9))
        n += 1
    return total / max(1, n)


def fit_temperature(records: list[dict], probs_key: str, true_key: str) -> float:
    """Coarse-to-fine search for the T minimising NLL. Returns T in [0.25, 10]."""
    best_t, best_nll = 1.0, float("inf")

    # Coarse pass, then refine around the winner.
    grid_hi = 20.0
    grid = [round(0.25 + 0.25 * i, 2) for i in range(int(grid_hi / 0.25))]  # 0.25 .. 20.0
    for t in grid:
        val = _nll(records, probs_key, true_key, t)
        if val < best_nll:
            best_nll, best_t = val, t

    fine = [round(best_t - 0.25 + 0.02 * i, 3) for i in range(25)]
    for t in fine:
        if t <= 0:
            continue
        val = _nll(records, probs_key, true_key, t)
        if val < best_nll:
            best_nll, best_t = val, t

    if best_t >= grid_hi - 0.25:
        # Hitting the ceiling means the head is so miscalibrated that softening
        # toward uniform keeps helping — worth flagging, not silently clipping.
        print(f"[gating] WARNING: {probs_key} T={best_t} hit the search ceiling; "
              f"confidence for this head is barely informative.")

    print(f"[gating] fit {probs_key}: T={best_t} (NLL {best_nll:.4f}, "
          f"vs T=1.0 NLL {_nll(records, probs_key, true_key, 1.0):.4f})")
    return best_t


def fit_and_save(records: list[dict]) -> dict:
    """Fit temperatures for department + urgency on cal records, save to JSON."""
    temps = {
        "department": fit_temperature(records, "department_probs", "true_department"),
        "urgency": fit_temperature(records, "urgency_probs", "true_urgency"),
    }
    TEMPERATURES_PATH.write_text(json.dumps(temps, indent=2))
    print(f"[gating] wrote {TEMPERATURES_PATH}: {temps}")
    return temps


def load_temperatures() -> dict:
    """Load fitted temperatures, or all-1.0 (no scaling) if not fit yet."""
    if TEMPERATURES_PATH.exists():
        return json.loads(TEMPERATURES_PATH.read_text())
    print("[gating] no temperatures.json yet -> using T=1.0 (uncalibrated)")
    return {"department": 1.0, "urgency": 1.0}


def calibrated_confidence(probs: dict, temperature: float) -> tuple[str, float]:
    """Return (argmax_label, confidence) after temperature scaling."""
    scaled = apply_temperature(probs, temperature)
    label = max(scaled, key=scaled.get)
    return label, scaled[label]


def _fit_on_cal_split():
    """Run the cal split through Laya, fit temperatures, print a before/after peek."""
    import pandas as pd
    from config import SPLITS_DIR
    from laya_pipeline.engine import load_router, run_split

    cal = pd.read_parquet(SPLITS_DIR / "cal.parquet")
    router = load_router()
    records, _ = run_split(router, cal)

    print("\n" + "=" * 60)
    print("STAGE 3 — TEMPERATURE CALIBRATION (fit on cal split)")
    print("=" * 60)
    temps = fit_and_save(records)

    # Peek: mean confidence before vs after, for both heads.
    for key, probs_key in (("department", "department_probs"), ("urgency", "urgency_probs")):
        before = sum(max(r[probs_key].values()) for r in records) / len(records)
        after = sum(
            max(apply_temperature(r[probs_key], temps[key]).values()) for r in records
        ) / len(records)
        print(f"[gating] {key}: mean top-prob {before:.3f} -> {after:.3f} after T={temps[key]}")
    print("=" * 60)


if __name__ == "__main__":
    _fit_on_cal_split()
