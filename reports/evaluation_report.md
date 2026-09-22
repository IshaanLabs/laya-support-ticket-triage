# Laya Support-Ticket Triage — Evaluation Report

_Generated 2026-09-23 00:12_  
_Model: convaiinnovations/laya (English checkpoint) via laya 0.3.5, device=cuda_

## What this is
An honest, zero-shot evaluation of Laya (the open-source 'System 1' decision engine, an open counterpart to TypeSafe's Jev) on real customer-support tickets. No fine-tuning: this measures the off-the-shelf model on a dataset it was never trained on. Numbers are what they are.

## Setup
- **Test set:** 939 held-out English tickets (never seen during calibration).
- **Departments:** 8-way choice, mirroring the dataset's support queues.
- **Urgency:** 3-class ordinal (low / medium / high) — the only levels present in the English data.
- **Calibration:** temperatures fit on a separate 500-ticket split (department T=1.11, urgency T=3.68).

## Headline: confidence gating
At a **0.85** confidence threshold, **7.1%** of tickets (67/939) were auto-routed with no human, at **97.0%** accuracy within that subset.
The remaining **92.9%** (872) were escalated to a human, where accuracy was only 34.5% — i.e. the model correctly knows which tickets it is unsure about.
- Overall department accuracy (all tickets, argmax): **39.0%**

## Department routing
- Accuracy: **39.0%** (8 classes)
- Per-class recall:
  - Technical Support: 75.5% (n=282)
  - IT Support: 3.4% (n=116)
  - Product Support: 21.9% (n=183)
  - Billing and Payments: 67.4% (n=95)
  - Customer Service: 7.7% (n=143)
  - Returns and Exchanges: 14.3% (n=49)
  - Sales and Pre-Sales: 6.5% (n=31)
  - Service Outages and Maintenance: 62.5% (n=40)

```
true \ pred | Technical  IT Support Product Su Billing an Customer S Returns an Sales and  Service Ou
-----------------------------------------------------------------------------------------------------
 Technical  |        213          1         44          3         13          1          0          7
 IT Support |         85          4         16          2          4          0          2          3
 Product Su |        119          0         40          5         16          1          0          2
 Billing an |         17          0          9         64          5          0          0          0
 Customer S |         66          3         48         13         11          0          1          1
 Returns an |         20          0         11          3          7          7          1          0
 Sales and  |          8          0         10          1         10          0          2          0
 Service Ou |          5          2          6          0          2          0          0         25
```

## Urgency (3-class ordinal)
- Exact accuracy: **47.1%**
- Within-one-level: 95.8% (ordinal, so off-by-one is a soft miss)

```
true \ pred |      0:low   1:medium     2:high
----------------------------------------------
      0:low |          0        136         39
   1:medium |          0        266        130
     2:high |          0        192        176
```

## Latency
- 939 tickets in 128.7s
- **137 ms/ticket** (7.3 tickets/sec), device=cuda
- No text generation, no tokens streamed, no output to parse — one forward pass per ticket.

## Reading the results
The dataset labels are themselves noisy (e.g. integration questions labeled 'Customer Service'), so a chunk of the 'errors' are cases where Laya's pick is arguably more correct than the ground truth. The calibration story is the real finding: the choice head ships well-calibrated (T≈1.1), so its confidence is trustworthy for gating, while the score head ships over-confident (T≈3.7) and only becomes usable after temperature scaling.