"""Stage 5: SQLite logging of every triaged ticket.

Plain functions over the stdlib `sqlite3` — no ORM, no classes. One table,
`tickets`, holding the input plus every prediction, confidence, the gate
decision, and (when known) whether it matched ground truth.

Backfill the 939-row eval from the parquet Stage 4 already produced:

    python -m storage.db --backfill

Then inspect / query:

    python -m storage.db --summary
    python -m storage.db --show escalate_to_human --limit 10
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime

from config import DB_PATH, REPORTS_DIR

_PRED_PATH = REPORTS_DIR / "predictions.parquet"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    source                TEXT NOT NULL,      -- 'eval' | 'live' | 'batch'
    text                  TEXT NOT NULL,
    pred_department       TEXT,
    department_confidence REAL,
    pred_urgency          INTEGER,
    urgency_confidence    REAL,
    churn_risk            REAL,
    refund_requested      REAL,
    gate_decision         TEXT,               -- 'auto_route' | 'escalate_to_human'
    true_department       TEXT,               -- NULL for live tickets w/o ground truth
    true_urgency          INTEGER,
    department_correct    INTEGER,            -- 1/0/NULL
    urgency_correct       INTEGER             -- 1/0/NULL
);
"""


def connect() -> sqlite3.Connection:
    """Open the DB (creating the table on first use). Row access by name."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def log_ticket(conn, *, text, source, pred_department, department_confidence,
               pred_urgency, urgency_confidence, churn_risk, refund_requested,
               gate_decision, true_department=None, true_urgency=None) -> int:
    """Insert one triaged ticket. Computes correctness when ground truth is given."""
    dept_correct = None
    urg_correct = None
    if true_department is not None:
        dept_correct = int(pred_department == true_department)
    if true_urgency is not None and pred_urgency is not None:
        urg_correct = int(int(pred_urgency) == int(true_urgency))

    cur = conn.execute(
        """INSERT INTO tickets (
            created_at, source, text, pred_department, department_confidence,
            pred_urgency, urgency_confidence, churn_risk, refund_requested,
            gate_decision, true_department, true_urgency,
            department_correct, urgency_correct
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"), source, text,
            pred_department, department_confidence, pred_urgency, urgency_confidence,
            churn_risk, refund_requested, gate_decision,
            true_department, true_urgency, dept_correct, urg_correct,
        ),
    )
    conn.commit()
    return cur.lastrowid


def backfill_from_eval(conn) -> int:
    """Load reports/predictions.parquet (the 939-row eval) into the DB.

    Idempotent-ish: clears prior 'eval' rows first so re-running doesn't dupe.
    """
    import pandas as pd

    if not _PRED_PATH.exists():
        print(f"[db] no predictions parquet at {_PRED_PATH}; run evaluation.run_eval first")
        return 0

    df = pd.read_parquet(_PRED_PATH)
    conn.execute("DELETE FROM tickets WHERE source = 'eval'")
    conn.commit()
    print(f"[db] backfilling {len(df)} eval rows ...")

    for _, r in df.iterrows():
        log_ticket(
            conn,
            text=r["text"],
            source="eval",
            pred_department=r["cal_department"],
            department_confidence=float(r["cal_department_confidence"]),
            pred_urgency=int(r["cal_urgency"]),
            urgency_confidence=float(r["cal_urgency_confidence"]),
            churn_risk=float(r["churn_risk"]),
            refund_requested=float(r["refund_requested"]),
            gate_decision=r["gate_decision"],
            true_department=r["true_department"],
            true_urgency=int(r["true_urgency"]),
        )
    print(f"[db] wrote {len(df)} rows to {DB_PATH}")
    return len(df)


def summary(conn) -> None:
    """Print a quick roll-up of what's logged."""
    total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    print(f"\n[db] {total} tickets logged in {DB_PATH}")
    if not total:
        return

    print("\nBy gate decision:")
    for row in conn.execute(
        """SELECT gate_decision, COUNT(*) n,
                  ROUND(AVG(department_correct)*100, 1) acc
           FROM tickets GROUP BY gate_decision"""
    ):
        acc = f"{row['acc']}%" if row["acc"] is not None else "n/a"
        print(f"   {row['gate_decision']:<20} n={row['n']:<5} dept_acc={acc}")

    print("\nBy predicted department:")
    for row in conn.execute(
        """SELECT pred_department, COUNT(*) n,
                  ROUND(AVG(department_correct)*100, 1) acc
           FROM tickets GROUP BY pred_department ORDER BY n DESC"""
    ):
        acc = f"{row['acc']}%" if row["acc"] is not None else "n/a"
        print(f"   {row['pred_department'] or '(none)':<32} n={row['n']:<5} acc={acc}")


def show(conn, gate_decision: str, limit: int = 10) -> None:
    """Print recent tickets with a given gate decision (e.g. escalations)."""
    print(f"\n[db] {gate_decision} tickets (up to {limit}):")
    for row in conn.execute(
        """SELECT id, pred_department, department_confidence, true_department, text
           FROM tickets WHERE gate_decision = ? ORDER BY id DESC LIMIT ?""",
        (gate_decision, limit),
    ):
        preview = row["text"][:70].replace("\n", " ")
        mark = "OK " if row["pred_department"] == row["true_department"] else "MISS"
        print(f"   #{row['id']} [{mark}] pred={row['pred_department']!r} "
              f"conf={row['department_confidence']:.2f} true={row['true_department']!r} :: {preview}...")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="load the eval parquet into the DB")
    ap.add_argument("--summary", action="store_true", help="print a roll-up")
    ap.add_argument("--show", metavar="DECISION", help="list tickets by gate decision")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    conn = connect()
    if args.backfill:
        backfill_from_eval(conn)
    if args.show:
        show(conn, args.show, args.limit)
    # Default to a summary if nothing else was asked.
    if args.summary or not (args.backfill or args.show):
        summary(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
