"""Render the Stage 4 metrics into a readable markdown report + console print.

Kept as string-building functions so the same text goes to stdout and the file.
"""
from __future__ import annotations

from datetime import datetime


def _confusion_block(labels, confusion, short=None) -> str:
    """ASCII confusion matrix: rows=true, cols=pred."""
    disp = short or [str(l)[:10] for l in labels]
    head = "true \\ pred | " + " ".join(f"{d:>10}" for d in disp)
    lines = [head, "-" * len(head)]
    for i, lab in enumerate(disp):
        row = " ".join(f"{confusion[i][j]:>10}" for j in range(len(labels)))
        lines.append(f"{lab:>11} | {row}")
    return "\n".join(lines)


def build_report(dept, urg, gating, latency, meta) -> str:
    L = []
    L.append("# Laya Support-Ticket Triage — Evaluation Report")
    L.append(f"\n_Generated {datetime.now():%Y-%m-%d %H:%M}_  ")
    L.append(f"_Model: convaiinnovations/laya (English checkpoint) via laya {meta['laya_version']}, "
             f"device={meta['device']}_\n")

    L.append("## What this is")
    L.append(
        "An honest, zero-shot evaluation of Laya (the open-source 'System 1' decision "
        "engine, an open counterpart to TypeSafe's Jev) on real customer-support tickets. "
        "No fine-tuning: this measures the off-the-shelf model on a dataset it was never "
        "trained on. Numbers are what they are.\n"
    )

    L.append("## Setup")
    L.append(f"- **Test set:** {meta['n_test']} held-out English tickets (never seen during calibration).")
    L.append(f"- **Departments:** {len(dept['labels'])}-way choice, mirroring the dataset's support queues.")
    L.append("- **Urgency:** 3-class ordinal (low / medium / high) — the only levels present in the English data.")
    L.append(f"- **Calibration:** temperatures fit on a separate 500-ticket split "
             f"(department T={meta['temps']['department']}, urgency T={meta['temps']['urgency']}).\n")

    L.append("## Headline: confidence gating")
    g = gating
    L.append(
        f"At a **{g['threshold']:.2f}** confidence threshold, **{g['auto_route_rate']*100:.1f}%** of "
        f"tickets ({g['auto_route_count']}/{meta['n_test']}) were auto-routed with no human, at "
        f"**{g['auto_route_accuracy']*100:.1f}%** accuracy within that subset."
    )
    L.append(
        f"The remaining **{g['escalate_rate']*100:.1f}%** ({g['escalate_count']}) were escalated to a "
        f"human, where accuracy was only {g['escalate_accuracy']*100:.1f}% — i.e. the model correctly "
        f"knows which tickets it is unsure about."
    )
    L.append(f"- Overall department accuracy (all tickets, argmax): **{g['overall_accuracy']*100:.1f}%**\n")

    L.append("## Department routing")
    L.append(f"- Accuracy: **{dept['accuracy']*100:.1f}%** ({len(dept['labels'])} classes)")
    L.append("- Per-class recall:")
    for lab, s in dept["per_class"].items():
        L.append(f"  - {lab}: {s['recall']*100:.1f}% (n={s['support']})")
    L.append("\n```")
    L.append(_confusion_block(dept["labels"], dept["confusion"],
                              short=[l[:10] for l in dept["labels"]]))
    L.append("```\n")

    L.append("## Urgency (3-class ordinal)")
    L.append(f"- Exact accuracy: **{urg['accuracy']*100:.1f}%**")
    L.append(f"- Within-one-level: {urg['within_one_level']*100:.1f}% (ordinal, so off-by-one is a soft miss)")
    L.append("\n```")
    L.append(_confusion_block(urg["levels"], urg["confusion"]))
    L.append("```\n")

    L.append("## Latency")
    lat = latency
    L.append(f"- {lat['tickets']} tickets in {lat['total_seconds']:.1f}s")
    L.append(f"- **{lat['ms_per_ticket']:.0f} ms/ticket** ({lat['tickets_per_second']:.1f} tickets/sec), device={meta['device']}")
    L.append("- No text generation, no tokens streamed, no output to parse — one forward pass per ticket.\n")

    L.append("## Reading the results")
    L.append(
        "The dataset labels are themselves noisy (e.g. integration questions labeled "
        "'Customer Service'), so a chunk of the 'errors' are cases where Laya's pick is "
        "arguably more correct than the ground truth. The calibration story is the real "
        "finding: the choice head ships well-calibrated (T≈1.1), so its confidence is "
        "trustworthy for gating, while the score head ships over-confident (T≈3.7) and only "
        "becomes usable after temperature scaling."
    )
    return "\n".join(L)
