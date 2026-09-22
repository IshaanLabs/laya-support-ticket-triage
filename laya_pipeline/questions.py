"""The typed questions we ask Laya about each ticket.

Built from config so the option/label space stays in one place. `department`
(choice) and `urgency` (score) are the two we score against ground truth;
`churn_risk` and `refund_requested` (noul) are live demo outputs only — the
dataset has no honest label for them, so we never claim accuracy on them.
"""
from __future__ import annotations

from config import DEPARTMENTS, URGENCY_RUBRIC


def build_questions() -> dict:
    """Return the Laya question dict mirroring our dataset label space."""
    return {
        "department": {
            "type": "choice",
            "instructions": "Which support department should handle this ticket?",
            "criteria": dict(DEPARTMENTS),  # 8 real support queues
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": list(URGENCY_RUBRIC),  # 0=very_low .. 4=critical
        },
        # --- demo-only, not scored ---
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the customer threaten to cancel, leave, or express strong dissatisfaction?",
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund, return, or money back?",
        },
    }


# Which answers we compare to ground truth vs. keep as demo signal only.
SCORED_QUESTIONS = ("department", "urgency")
DEMO_QUESTIONS = ("churn_risk", "refund_requested")
