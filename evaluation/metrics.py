"""Stage 4 metrics: scoring Laya's predictions against ground truth.

Plain functions over the list-of-record dicts that engine.run_split produces.
Temperatures (Stage 3) are applied here so every reported confidence is the
calibrated one. The headline number is the auto-route rate and the accuracy
within that auto-routed subset.
"""
from __future__ import annotations

from config import CONFIDENCE_THRESHOLD, URGENCY_RUBRIC
from data.labels import DEPARTMENT_LABELS
from laya_pipeline.gating import apply_temperature, calibrated_confidence, gate


def _accuracy(pairs: list[tuple]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def apply_calibration(records: list[dict], temps: dict) -> list[dict]:
    """Recompute department/urgency prediction + confidence with fitted temps.

    Department: argmax can shift slightly after scaling; we take the calibrated
    argmax + confidence. Urgency: keep the expected-value class but replace the
    confidence with the calibrated top-prob (the raw one was over-confident).
    """
    t_dept = temps.get("department", 1.0)
    t_urg = temps.get("urgency", 1.0)
    for r in records:
        dept_label, dept_conf = calibrated_confidence(r["department_probs"], t_dept)
        r["cal_department"] = dept_label
        r["cal_department_confidence"] = dept_conf

        # Urgency is ORDINAL. The PREDICTION comes from the RAW distribution's
        # expected value (Laya's own `score`): a big softening temperature drags
        # every expected value toward the middle and would collapse all tickets
        # to "medium". The CONFIDENCE, by contrast, uses the temperature-scaled
        # distribution (raw is over-confident). Two goals, two distributions.
        raw = r["urgency_probs"]
        raw_exp = sum(int(k) * p for k, p in raw.items())
        urg_class = max(0, min(int(round(raw_exp)), len(raw) - 1))
        urg_scaled = apply_temperature(raw, t_urg)
        r["cal_urgency"] = urg_class
        r["cal_urgency_confidence"] = urg_scaled.get(str(urg_class), 0.0)
    return records


def department_metrics(records: list[dict]) -> dict:
    """Accuracy + confusion matrix for the department choice."""
    labels = list(DEPARTMENT_LABELS)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    confusion = [[0] * n for _ in range(n)]
    pairs = []
    for r in records:
        true, pred = r["true_department"], r["cal_department"]
        pairs.append((true, pred))
        if true in idx and pred in idx:
            confusion[idx[true]][idx[pred]] += 1
    # Per-class recall for readability.
    per_class = {}
    for lab in labels:
        rows = [(t, p) for t, p in pairs if t == lab]
        per_class[lab] = {"support": len(rows), "recall": _accuracy(rows)}
    return {
        "accuracy": _accuracy(pairs),
        "labels": labels,
        "confusion": confusion,
        "per_class": per_class,
    }


def urgency_metrics(records: list[dict]) -> dict:
    """3-class accuracy + confusion for urgency (0=low,1=med,2=high)."""
    n = len(URGENCY_RUBRIC)  # 3
    confusion = [[0] * n for _ in range(n)]
    pairs = []
    for r in records:
        true, pred = int(r["true_urgency"]), int(r["cal_urgency"])
        pairs.append((true, pred))
        if 0 <= true < n and 0 <= pred < n:
            confusion[true][pred] += 1
    # Off-by-one accuracy: ordinal, so being 1 level off is a softer error.
    within_one = _accuracy([(0, 0) if abs(t - p) <= 1 else (0, 1) for t, p in pairs])
    return {
        "accuracy": _accuracy(pairs),
        "within_one_level": within_one,
        "levels": [f"{i}:{URGENCY_RUBRIC[i].split(':')[0]}" for i in range(n)],
        "confusion": confusion,
    }


def gating_metrics(records: list[dict], threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """The headline: how many tickets clear the threshold, and how accurate they are.

    Computed on the department decision (the routing action). Auto-routed = the
    tickets we'd send straight to a queue with no human; the interesting number
    is accuracy inside that confident subset vs. the escalated remainder.
    """
    auto, escalated = [], []
    for r in records:
        decision = gate(r["cal_department_confidence"], threshold)
        r["gate_decision"] = decision
        (auto if decision == "auto_route" else escalated).append(r)

    def acc(rows):
        return _accuracy([(x["true_department"], x["cal_department"]) for x in rows])

    total = len(records)
    return {
        "threshold": threshold,
        "auto_route_count": len(auto),
        "auto_route_rate": len(auto) / total if total else 0.0,
        "auto_route_accuracy": acc(auto),
        "escalate_count": len(escalated),
        "escalate_rate": len(escalated) / total if total else 0.0,
        "escalate_accuracy": acc(escalated),
        "overall_accuracy": acc(records),
    }


def latency_metrics(records: list[dict], total_seconds: float) -> dict:
    n = max(1, len(records))
    return {
        "tickets": len(records),
        "total_seconds": total_seconds,
        "ms_per_ticket": total_seconds / n * 1000,
        "tickets_per_second": n / total_seconds if total_seconds else 0.0,
    }
