"""Load the Laya router once and run typed questions over tickets.

Simple functions, no classes — the router object comes from the laya library
and we just pass it around. English-only preload keeps us inside 4GB VRAM.

Quick manual sanity check (Stage 2):

    python -m laya_pipeline.engine
"""
from __future__ import annotations

import os
import time

from config import BATCH_SIZE, LAYA_CHECKPOINTS, get_device
from laya_pipeline.questions import build_questions

# transformers probes TensorFlow at import; when TF is present its abseil
# runtime can deadlock Laya's model build. The README says: run with USE_TF=0.
os.environ.setdefault("USE_TF", "0")


def load_router():
    """Build the Laya router with only the English checkpoint preloaded."""
    from laya import Router

    device = get_device()
    print(f"[engine] loading Laya router on device={device}, checkpoints={LAYA_CHECKPOINTS} ...")
    t0 = time.time()
    router = Router(preload=False, device=device)
    router.preload(LAYA_CHECKPOINTS)  # english only -> ~808MB weights, fits 4GB
    print(f"[engine] router ready in {time.time() - t0:.1f}s")
    return router


def _to_state(text: str) -> dict:
    """Laya accepts a dict state; wrap the ticket text in a 'body' field."""
    return {"body": text}


def predict_one(router, text: str, questions: dict | None = None) -> dict:
    """Run all questions over one ticket. Returns Laya's raw result dict."""
    questions = questions or build_questions()
    return router.predict(_to_state(text), questions)


def predict_batch(router, texts: list[str], questions: dict | None = None,
                  batch_size: int = BATCH_SIZE) -> list[dict]:
    """Run questions over many tickets in batches (much faster per ticket).

    Falls back to one-at-a-time if the router has no batch predict method, so
    this works regardless of the installed laya version.
    """
    questions = questions or build_questions()
    states = [_to_state(t) for t in texts]
    results: list[dict] = []

    has_batch = hasattr(router, "predict_batch")
    for i in range(0, len(states), batch_size):
        chunk = states[i:i + batch_size]
        if has_batch:
            results.extend(router.predict_batch(chunk, questions))
        else:
            results.extend(router.predict(s, questions) for s in chunk)
        print(f"[engine] predicted {min(i + batch_size, len(states))}/{len(states)}")
    return results


def summarize(result: dict) -> dict:
    """Pull the fields we care about out of Laya's result into a flat dict.

    `urgency_score` is Laya's continuous expected value over the 3-level rubric
    (0=low, 1=medium, 2=high); `urgency_class` rounds it to the nearest level so
    it can be compared to the dataset's 0..2 ground truth.
    """
    ans = result["answers"]
    raw_score = ans["urgency"]["score"]
    urgency_class = int(round(raw_score))
    urgency_class = max(0, min(urgency_class, 2))  # clamp into 0..2
    return {
        "department": ans["department"]["choice"],
        "department_confidence": ans["department"]["confidence"],
        "urgency_score": raw_score,
        "urgency_class": urgency_class,
        "urgency_confidence": ans["urgency"]["confidence"],
        "churn_risk": ans["churn_risk"]["noul"],
        "refund_requested": ans["refund_requested"]["noul"],
        "routed_model": result.get("routing", {}).get("model", "?"),
    }


def run_split(router, df, questions: dict | None = None):
    """Run every ticket in a split DataFrame through Laya.

    Returns (records, total_seconds). Each record keeps the summarized fields
    plus the raw probability dicts (needed for temperature calibration) and the
    ground-truth columns so downstream stages don't have to re-join.
    """
    questions = questions or build_questions()
    texts = df["text"].tolist()
    print(f"[engine] running {len(texts)} tickets through Laya ...")

    t0 = time.time()
    results = predict_batch(router, texts, questions)
    total_s = time.time() - t0
    print(f"[engine] done in {total_s:.1f}s ({total_s / max(1, len(texts)) * 1000:.0f} ms/ticket avg)")

    records = []
    df = df.reset_index(drop=True)
    for i, result in enumerate(results):
        s = summarize(result)
        ans = result["answers"]
        records.append({
            "text": texts[i],
            "true_department": df.at[i, "department"],
            "true_urgency": int(df.at[i, "urgency_score"]),
            "pred_department": s["department"],
            "department_confidence": s["department_confidence"],
            "department_probs": ans["department"]["probabilities"],
            "pred_urgency": s["urgency_class"],
            "urgency_score_raw": s["urgency_score"],
            "urgency_confidence": s["urgency_confidence"],
            "urgency_probs": ans["urgency"]["probabilities"],
            "churn_risk": s["churn_risk"],
            "refund_requested": s["refund_requested"],
        })
    return records, total_s


def _sanity_check():
    """Load a few tickets from the test split and print predictions to eyeball."""
    import pandas as pd
    from config import SPLITS_DIR

    df = pd.read_parquet(SPLITS_DIR / "test.parquet").head(5)
    router = load_router()

    print("\n" + "=" * 70)
    print("STAGE 2 SANITY CHECK — 5 test tickets")
    print("=" * 70)
    for _, row in df.iterrows():
        t0 = time.time()
        result = predict_one(router, row["text"])
        ms = (time.time() - t0) * 1000
        s = summarize(result)
        preview = row["text"][:80].replace("\n", " ")
        print(f"\nTICKET: {preview}...")
        print(f"  truth    -> dept={row['department']!r}  urgency={row['urgency_score']} (0=low,1=med,2=high)")
        print(f"  laya     -> dept={s['department']!r} (conf {s['department_confidence']:.2f})"
              f"  urgency={s['urgency_class']} raw={s['urgency_score']:.2f} (conf {s['urgency_confidence']:.2f})")
        print(f"  demo     -> churn={s['churn_risk']:.2f}  refund={s['refund_requested']:.2f}")
        print(f"  routed={s['routed_model']}  latency={ms:.0f}ms")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    _sanity_check()
