"""Runnable checks for the metrics logic on synthetic records (no GPU/Laya).
Run: python -m evaluation.check_metrics
"""
from data.labels import DEPARTMENT_LABELS
from evaluation.metrics import (
    apply_calibration,
    department_metrics,
    gating_metrics,
    urgency_metrics,
)

d0, d1 = DEPARTMENT_LABELS[0], DEPARTMENT_LABELS[1]

# 4 synthetic records: 2 confident+correct, 1 confident+wrong, 1 unsure+wrong.
records = [
    {"true_department": d0, "department_probs": {d0: 0.95, d1: 0.05},
     "true_urgency": 2, "urgency_probs": {"0": 0.1, "1": 0.2, "2": 0.7}},
    {"true_department": d0, "department_probs": {d0: 0.90, d1: 0.10},
     "true_urgency": 1, "urgency_probs": {"0": 0.2, "1": 0.7, "2": 0.1}},
    {"true_department": d0, "department_probs": {d1: 0.92, d0: 0.08},  # confident but WRONG
     "true_urgency": 0, "urgency_probs": {"0": 0.2, "1": 0.5, "2": 0.3}},  # exp=1.1 -> 1, truth 0: off-by-1 miss
    {"true_department": d0, "department_probs": {d0: 0.40, d1: 0.60},  # unsure + wrong
     "true_urgency": 2, "urgency_probs": {"0": 0.1, "1": 0.2, "2": 0.7}},  # exp=1.6 -> 2, correct
]

# No calibration (T=1) so we can reason about raw numbers.
apply_calibration(records, {"department": 1.0, "urgency": 1.0})

dept = department_metrics(records)
# 2 of 4 department predictions correct (records 0 and 1).
assert abs(dept["accuracy"] - 0.5) < 1e-9, dept["accuracy"]
# Confusion diagonal for d0 row: 2 correct.
assert dept["confusion"][0][0] == 2

urg = urgency_metrics(records)
# urgency correct: rec0(2), rec1(1), rec3(2) = 3/4; rec2 predicts 1 but truth 0.
assert abs(urg["accuracy"] - 0.75) < 1e-9, urg["accuracy"]
# within-one: rec2 is off by 1 (pred1,true0) -> counts as within-one -> all 4.
assert abs(urg["within_one_level"] - 1.0) < 1e-9

g = gating_metrics(records, threshold=0.85)
# Confident (>=0.85): recs 0,1,2 -> 3 auto-routed; rec3 (0.60) escalated.
assert g["auto_route_count"] == 3
assert g["escalate_count"] == 1
# Within auto-routed: recs 0,1 correct, rec2 wrong -> 2/3.
assert abs(g["auto_route_accuracy"] - 2 / 3) < 1e-6, g["auto_route_accuracy"]
# Escalated (rec3) was wrong -> 0.0 accuracy in that bucket.
assert g["escalate_accuracy"] == 0.0

print("OK: metrics checks passed")
