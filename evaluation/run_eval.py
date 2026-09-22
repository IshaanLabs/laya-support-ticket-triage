"""Stage 4 entrypoint: full evaluation on the held-out test set.

    python -m evaluation.run_eval             # full 939-row test set
    python -m evaluation.run_eval --limit 50  # quick smoke run

Runs the test split through Laya once, applies the fitted temperatures, computes
department / urgency / gating / latency metrics, prints them, writes a markdown
report to reports/, and dumps raw predictions to reports/predictions.parquet so
Stages 5-6 reuse them without re-running the model.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import laya as laya_lib
from config import REPORTS_DIR, SPLITS_DIR, get_device
from evaluation.metrics import (
    apply_calibration,
    department_metrics,
    gating_metrics,
    latency_metrics,
    urgency_metrics,
)
from evaluation.report import build_report
from laya_pipeline.engine import load_router, run_split
from laya_pipeline.gating import load_temperatures

PRED_PATH = REPORTS_DIR / "predictions.parquet"
REPORT_PATH = REPORTS_DIR / "evaluation_report.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only evaluate the first N test rows")
    args = ap.parse_args()

    test = pd.read_parquet(SPLITS_DIR / "test.parquet")
    if args.limit:
        test = test.head(args.limit)
    print(f"[eval] test rows: {len(test)}")

    temps = load_temperatures()
    print(f"[eval] using temperatures: {temps}")

    router = load_router()
    records, total_s = run_split(router, test)
    apply_calibration(records, temps)

    dept = department_metrics(records)
    urg = urgency_metrics(records)
    gating = gating_metrics(records)
    latency = latency_metrics(records, total_s)

    meta = {
        "n_test": len(records),
        "device": get_device(),
        "laya_version": getattr(laya_lib, "__version__", "?"),
        "temps": temps,
    }

    report = build_report(dept, urg, gating, latency, meta)
    print("\n" + report + "\n")
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[eval] wrote report -> {REPORT_PATH}")

    # Persist raw predictions (drop the bulky prob dicts to JSON strings for parquet).
    import json

    df = pd.DataFrame(records)
    for col in ("department_probs", "urgency_probs"):
        df[col] = df[col].map(lambda d: json.dumps(d))
    df.to_parquet(PRED_PATH, index=False)
    print(f"[eval] wrote predictions -> {PRED_PATH} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
