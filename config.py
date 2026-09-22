"""Central configuration and shared helpers.

Kept deliberately small: paths, VRAM-conservative Laya defaults, the label
spaces, and a device auto-detector. Everything else imports from here so the
tuning knobs live in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # raw HF download + cleaned parquet
SPLITS_DIR = DATA_DIR / "splits"        # cal / dev / test parquet files
REPORTS_DIR = ROOT / "reports"
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "triage.db"

for _d in (DATA_DIR, CACHE_DIR, SPLITS_DIR, REPORTS_DIR, STORAGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
HF_DATASET = "Tobi-Bueck/customer-support-tickets"
# The repo ships 3 CSVs with different schemas; auto-merge fails. Pin the one
# full-schema file (subject/body/queue/priority/language/type/tags) we inspected.
HF_DATA_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
DATASET_LANGUAGE = "en"                 # English-only: single checkpoint, fits 4GB VRAM

# 3-way split so the eval stays honest: fit calibration on `cal`, pick the
# gating threshold on `dev`, report the headline number on untouched `test`.
SPLIT_SIZES = {"cal": 500, "dev": 500, "test": 1000}
SPLIT_SEED = 42

# --------------------------------------------------------------------------
# Label spaces (mirror the dataset so prediction space == ground-truth space)
# --------------------------------------------------------------------------
# The dataset's `queue` column mixes 10 real support queues with ~40 website
# categories. We keep the genuine support departments and drop the rest.
DEPARTMENTS: dict[str, str] = {
    "Technical Support": "software bugs, device errors, connectivity, troubleshooting",
    "IT Support": "internal IT, accounts, office equipment, network access",
    "Product Support": "product features, setup, integration, how-to questions",
    "Billing and Payments": "invoices, charges, payments, billing disputes",
    "Customer Service": "general account help, information requests, inquiries",
    "Returns and Exchanges": "returns, refunds, exchanges, product replacement",
    "Sales and Pre-Sales": "pricing, new contracts, product evaluation before buying",
    "Service Outages and Maintenance": "outages, downtime, scheduled maintenance windows",
}
# Queues present in the data that we deliberately exclude (tiny / not triage targets).
DROPPED_QUEUES = {"Human Resources", "General Inquiry"}

# The English rows in this CSV only carry 3 priority levels (low/medium/high),
# so we use a 3-level rubric. Laya returns `score` as an expected value over the
# rubric indexed from 0, i.e. 0=low, 1=medium, 2=high. We map dataset priority
# to that same 0..2 index (see data/labels.py) so score and truth line up.
PRIORITY_TO_SCORE: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}
SCORE_TO_PRIORITY = {v: k for k, v in PRIORITY_TO_SCORE.items()}
URGENCY_RUBRIC = [
    "low: routine request or general question, no time pressure",  # 0
    "medium: a normal issue affecting the user",                    # 1
    "high: important, user is blocked, impacted, or urgent",        # 2
]
# Laya's score is continuous; round to nearest rubric index to get a class.
URGENCY_LEVELS = len(URGENCY_RUBRIC)  # 3

# --------------------------------------------------------------------------
# Laya runtime defaults (tuned for a 4GB RTX 3050)
# --------------------------------------------------------------------------
# English-only means we preload just the one checkpoint instead of all three,
# which is what keeps us inside 4GB of VRAM.
LAYA_CHECKPOINTS = ["english"]
BATCH_SIZE = 8                          # lower to 4 if you hit CUDA OOM
CONFIDENCE_THRESHOLD = 0.85             # auto-route vs escalate; docs' default


def get_device() -> str:
    """Return 'cuda' when a GPU is usable, else 'cpu'. Never raises."""
    if os.environ.get("LAYA_FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
