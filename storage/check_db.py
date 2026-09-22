"""Runnable check for the DB logging logic (insert + correctness + query).
Uses a throwaway sqlite file so it never touches the real triage.db.
Run: python -m storage.check_db
"""
import sqlite3
import tempfile
from pathlib import Path

from storage import db as dbmod

# Point the module at a temp DB file for the duration of the check.
_tmp = Path(tempfile.gettempdir()) / "laya_check_triage.db"
if _tmp.exists():
    _tmp.unlink()
dbmod.DB_PATH = _tmp

conn = dbmod.connect()

# 1. Ground-truth match -> department_correct = 1, urgency wrong -> 0.
rid = dbmod.log_ticket(
    conn, text="billing issue", source="eval",
    pred_department="Billing and Payments", department_confidence=0.99,
    pred_urgency=2, urgency_confidence=0.5, churn_risk=0.1, refund_requested=0.2,
    gate_decision="auto_route",
    true_department="Billing and Payments", true_urgency=1,
)
row = conn.execute("SELECT * FROM tickets WHERE id=?", (rid,)).fetchone()
assert row["department_correct"] == 1
assert row["urgency_correct"] == 0
assert row["gate_decision"] == "auto_route"

# 2. Wrong department -> 0.
dbmod.log_ticket(
    conn, text="tech issue", source="eval",
    pred_department="Technical Support", department_confidence=0.4,
    pred_urgency=1, urgency_confidence=0.5, churn_risk=0.0, refund_requested=0.0,
    gate_decision="escalate_to_human",
    true_department="IT Support", true_urgency=1,
)

# 3. Live ticket with NO ground truth -> correctness columns stay NULL.
dbmod.log_ticket(
    conn, text="live one", source="live",
    pred_department="Customer Service", department_confidence=0.7,
    pred_urgency=0, urgency_confidence=0.6, churn_risk=0.0, refund_requested=0.0,
    gate_decision="escalate_to_human",
)
live = conn.execute("SELECT * FROM tickets WHERE source='live'").fetchone()
assert live["department_correct"] is None
assert live["true_department"] is None

# 4. Counts + a grouped query behave.
total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
assert total == 3
escalated = conn.execute(
    "SELECT COUNT(*) FROM tickets WHERE gate_decision='escalate_to_human'"
).fetchone()[0]
assert escalated == 2

conn.close()
_tmp.unlink()
print("OK: sqlite logging checks passed")
