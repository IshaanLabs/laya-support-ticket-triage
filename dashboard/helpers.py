"""Shared dashboard helpers: cached router + one-ticket triage.

Kept as plain functions. The router load is ~45s so it's cached with
st.cache_resource; everything else is cheap. Reused by the Live and Batch tabs.
"""
from __future__ import annotations

import json
import time

import streamlit as st

from config import CONFIDENCE_THRESHOLD
from laya_pipeline.engine import load_router, predict_one, summarize
from laya_pipeline.gating import apply_temperature, calibrated_confidence, gate, load_temperatures


@st.cache_resource(show_spinner="Loading Laya (first time ~45s, then cached)...")
def get_router():
    """Load the Laya router once per Streamlit session."""
    print("[dashboard] loading router (cached)...")
    return load_router()


@st.cache_data
def get_temperatures() -> dict:
    return load_temperatures()


def triage_ticket(router, text: str, temps: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Run one ticket through our 8-category questions, apply calibration, gate.

    Returns a flat dict ready to display and to log to SQLite.
    """
    t0 = time.time()
    result = predict_one(router, text)
    latency_ms = (time.time() - t0) * 1000
    s = summarize(result)

    # Calibrated department confidence (T=1.11) drives the gate.
    dept_label, dept_conf = calibrated_confidence(
        result["answers"]["department"]["probabilities"], temps.get("department", 1.0)
    )
    # Urgency: raw expected value for the class, scaled distribution for confidence.
    urg_probs = result["answers"]["urgency"]["probabilities"]
    urg_scaled = apply_temperature(urg_probs, temps.get("urgency", 1.0))
    urg_conf = urg_scaled.get(str(s["urgency_class"]), 0.0)

    return {
        "text": text,
        "department": dept_label,
        "department_confidence": dept_conf,
        "department_probs": result["answers"]["department"]["probabilities"],
        "urgency_class": s["urgency_class"],
        "urgency_label": ["low", "medium", "high"][s["urgency_class"]],
        "urgency_confidence": urg_conf,
        "churn_risk": s["churn_risk"],
        "refund_requested": s["refund_requested"],
        "gate_decision": gate(dept_conf, threshold),
        "latency_ms": latency_ms,
    }


def run_builtin_preset(router, text: str) -> dict:
    """Run Laya's OWN built-in triage_questions() preset — zero config from us.

    Showcases 'pure Laya' out of the box. We render whatever it returns
    generically so we don't assume its exact schema.
    """
    import laya

    questions = laya.triage_questions()
    # The preset's instructions reference a `message` field, so provide it there.
    result = router.predict({"message": text}, questions)
    return {"questions": questions, "answers": result.get("answers", {})}


def interesting_cases(df, n: int = 6):
    """Pull illustrative rows from the eval predictions for the report tab.

    Returns (confident_correct, disagreements) — the first shows the model at
    its best, the second shows where it confidently disagrees with the (noisy)
    dataset label, which is often a labeling-quality story rather than a model
    error.
    """
    correct = df[(df["cal_department"] == df["true_department"])
                 & (df["cal_department_confidence"] >= 0.85)]
    confident_correct = correct.sort_values("cal_department_confidence", ascending=False).head(n)
    wrong = df[(df["cal_department"] != df["true_department"])
               & (df["cal_department_confidence"] >= 0.6)]
    disagreements = wrong.sort_values("cal_department_confidence", ascending=False).head(n)
    return confident_correct, disagreements


def flatten_answer(ans: dict) -> str:
    """Human-readable one-liner for any Laya answer dict (choice/score/noul)."""
    kind = ans.get("type")
    if kind == "choice":
        return f"{ans.get('choice')}  (conf {ans.get('confidence', 0):.2f})"
    if kind == "score":
        return f"score {ans.get('score', 0):.2f}  (conf {ans.get('confidence', 0):.2f})"
    if kind == "noul":
        return f"{ans.get('noul', 0):.2f} probability"
    # Fallback: show the raw dict compactly.
    return json.dumps({k: v for k, v in ans.items() if k != "action"}, default=str)
