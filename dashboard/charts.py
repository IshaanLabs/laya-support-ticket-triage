"""Matplotlib charts for the Evaluation tab. Plain functions returning figures.

All figures share one compact size and style so the Evaluation tab reads as a
tidy, consistent dashboard rather than three mismatched plots. Reads the Stage 4
predictions parquet; no Laya / no GPU needed.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt
import pandas as pd

from config import CONFIDENCE_THRESHOLD, REPORTS_DIR
from data.labels import DEPARTMENT_LABELS

_PRED_PATH = REPORTS_DIR / "predictions.parquet"

# One shared look for every chart -> consistent, dashboard-like.
FIGSIZE = (5.2, 3.6)
DPI = 130
TITLE_SIZE = 10
LABEL_SIZE = 8
TICK_SIZE = 7

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": TICK_SIZE,
})


def _new_fig():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    return fig, ax


def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(_PRED_PATH)
    for col in ("department_probs", "urgency_probs"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


def department_confusion_fig(df: pd.DataFrame):
    labels = list(DEPARTMENT_LABELS)
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for _, r in df.iterrows():
        t, p = r["true_department"], r["cal_department"]
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1

    fig, ax = _new_fig()
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    short = [l[:12] for l in labels]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Department confusion")
    peak = max(max(r) for r in mat) or 1
    for i in range(n):
        for j in range(n):
            if mat[i][j]:
                ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=6,
                        color="white" if mat[i][j] > peak / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def urgency_confusion_fig(df: pd.DataFrame):
    levels = ["low", "medium", "high"]
    mat = [[0] * 3 for _ in range(3)]
    for _, r in df.iterrows():
        t, p = int(r["true_urgency"]), int(r["cal_urgency"])
        if 0 <= t < 3 and 0 <= p < 3:
            mat[t][p] += 1

    fig, ax = _new_fig()
    im = ax.imshow(mat, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(levels)
    ax.set_yticks(range(3)); ax.set_yticklabels(levels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Urgency confusion")
    peak = max(max(r) for r in mat) or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, mat[i][j], ha="center", va="center", fontsize=9,
                    color="white" if mat[i][j] > peak / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def threshold_tradeoff_fig(df: pd.DataFrame):
    """Auto-route rate and within-auto-route accuracy as the threshold sweeps."""
    thresholds = [i / 100 for i in range(50, 100, 2)]  # 0.50 .. 0.98
    rates, accs = [], []
    total = len(df)
    for th in thresholds:
        auto = df[df["cal_department_confidence"] >= th]
        rates.append(len(auto) / total * 100 if total else 0)
        accs.append((auto["cal_department"] == auto["true_department"]).mean() * 100
                    if len(auto) else float("nan"))

    fig, ax1 = _new_fig()
    ax1.plot(thresholds, rates, "o-", color="tab:blue", ms=3, lw=1.4, label="auto-route rate")
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("auto-route rate (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.plot(thresholds, accs, "s-", color="tab:green", ms=3, lw=1.4, label="acc within auto-routed")
    ax2.set_ylabel("accuracy (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(0, 105)

    ax1.axvline(CONFIDENCE_THRESHOLD, color="grey", ls="--", lw=1, alpha=0.7)
    ax1.set_title("Coverage vs. accuracy")
    return fig
