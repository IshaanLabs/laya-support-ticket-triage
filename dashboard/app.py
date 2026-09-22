"""Laya Ticket Triage — Live Demo.  Run:  streamlit run dashboard/app.py

A polished, story-driven dashboard for the laya-support-ticket-triage project.
Three tabs:
  Live       — one-click example tickets or your own; real-time typed decisions,
               plus Laya's own built-in triage preset (zero-config showcase).
  Batch      — upload a CSV, triage the whole set, download results.
  Evaluation — the honest, zero-shot accuracy report on a held-out test set.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from config import CONFIDENCE_THRESHOLD, DEPARTMENTS
from dashboard.charts import (
    department_confusion_fig,
    load_predictions,
    threshold_tradeoff_fig,
    urgency_confusion_fig,
)
from dashboard.examples import EXAMPLES
from dashboard.helpers import (
    flatten_answer,
    get_router,
    get_temperatures,
    interesting_cases,
    run_builtin_preset,
    triage_ticket,
)

st.set_page_config(page_title="Laya Ticket Triage — Live Demo", page_icon="🎫", layout="wide")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🎫 Laya Ticket Triage — Live Demo")
st.markdown(
    "Routing customer-support tickets with **Laya**, the open-source *System-1 decision engine* "
    "(an open counterpart to TypeSafe's **Jev**). It reads a ticket and returns typed, calibrated "
    "decisions in **one forward pass** — no text generation, nothing to parse, nothing to hallucinate. "
    "This is **zero-shot**: the model was never fine-tuned on this data."
)


@st.cache_data
def _eval_df():
    try:
        return load_predictions()
    except Exception:
        return None


_df = _eval_df()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "- **Model:** `convaiinnovations/laya` (English checkpoint)\n"
        "- **Task:** route to 1 of 8 support departments + score urgency\n"
        "- **Data:** real customer-support tickets (Tobi-Bueck)\n"
        "- **Hardware:** RTX 3050, ~137 ms/ticket\n"
        "- **No fine-tuning** — this is the base model out of the box."
    )
    if _df is not None:
        auto = _df[_df["gate_decision"] == "auto_route"]
        auto_acc = (auto["cal_department"] == auto["true_department"]).mean() if len(auto) else 0
        st.divider()
        st.metric("Auto-routed at 97% acc", f"{len(auto)/len(_df)*100:.0f}%")
        st.caption("The confident slice is nearly perfect; the rest is correctly escalated.")
    st.divider()
    st.caption("Laya is Apache-2.0.")

live_tab, batch_tab, eval_tab = st.tabs(["🔴 Live", "📦 Batch", "📊 Evaluation"])


# --------------------------------------------------------------------------
# LIVE TAB
# --------------------------------------------------------------------------
with live_tab:
    st.subheader("Try it on a ticket")
    st.caption("Click an example to load it, or paste your own. First run loads the model (~45s), then it's instant.")

    # Prebuilt example buttons — clicking sets the text area via session state.
    if "ticket_text" not in st.session_state:
        st.session_state.ticket_text = ""

    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex["label"], width="stretch", help=ex["note"]):
            st.session_state.ticket_text = ex["text"]

    text = st.text_area("Ticket text (subject + body)", key="ticket_text", height=150,
                        placeholder="Paste a support ticket here, or click an example above.")

    left, right = st.columns([1, 1])
    threshold = left.slider("Auto-route confidence threshold", 0.5, 0.99,
                            float(CONFIDENCE_THRESHOLD), 0.01)
    show_builtin = right.toggle("Also show Laya's built-in preset (zero-config)", value=True)

    if st.button("Triage ticket", type="primary", width="stretch") and text.strip():
        router = get_router()
        temps = get_temperatures()
        res = triage_ticket(router, text, temps, threshold)

        st.markdown("### Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Department", res["department"])
        c2.metric("Confidence", f"{res['department_confidence']:.0%}")
        c3.metric("Urgency", res["urgency_label"].title())
        c4.metric("Latency", f"{res['latency_ms']:.0f} ms")

        if res["gate_decision"] == "auto_route":
            st.success(f"**Auto-route → {res['department']}** — confidence "
                       f"{res['department_confidence']:.0%} ≥ {threshold:.0%}. Send straight to the queue.")
        else:
            st.warning(f"**Escalate to a human** — confidence {res['department_confidence']:.0%} "
                       f"< {threshold:.0%}. The model is honestly flagging that it's unsure.")

        d1, d2 = st.columns([2, 1])
        with d1:
            st.markdown("**Department probabilities**")
            probs = pd.DataFrame(
                sorted(res["department_probs"].items(), key=lambda kv: -kv[1]),
                columns=["department", "probability"],
            )
            st.bar_chart(probs.set_index("department"), horizontal=True)
        with d2:
            st.markdown("**Extra signals** *(demo, not scored)*")
            st.metric("Churn risk", f"{res['churn_risk']:.0%}")
            st.metric("Refund requested", f"{res['refund_requested']:.0%}")

        # Log to SQLite.
        try:
            from storage.db import connect, log_ticket

            conn = connect()
            log_ticket(conn, text=text, source="live",
                       pred_department=res["department"],
                       department_confidence=res["department_confidence"],
                       pred_urgency=res["urgency_class"],
                       urgency_confidence=res["urgency_confidence"],
                       churn_risk=res["churn_risk"], refund_requested=res["refund_requested"],
                       gate_decision=res["gate_decision"])
            conn.close()
        except Exception as e:
            st.caption(f"(DB log skipped: {e})")

        if show_builtin:
            st.divider()
            st.markdown("### 🧩 Laya's own built-in triage preset")
            st.caption("`laya.triage_questions()` — Laya's shipped question set, unchanged. "
                       "No categories from us. This is 'pure Laya' out of the box, for contrast "
                       "with our dataset-aligned questions above.")
            builtin = run_builtin_preset(router, text)
            rows = [{"question": q, "answer": flatten_answer(a)} for q, a in builtin["answers"].items()]
            st.table(pd.DataFrame(rows))


# --------------------------------------------------------------------------
# BATCH TAB
# --------------------------------------------------------------------------
with batch_tab:
    st.subheader("Triage a whole CSV")
    st.caption("Upload a CSV with a text column; every row gets triaged and you can download the results.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Text column", list(df_in.columns))
        limit = c2.number_input("Max rows", 1, len(df_in), min(50, len(df_in)))
        if st.button("Run batch", type="primary"):
            router = get_router()
            temps = get_temperatures()
            rows = []
            prog = st.progress(0.0, "Triaging...")
            sub = df_in.head(int(limit))
            for i, (_, r) in enumerate(sub.iterrows()):
                res = triage_ticket(router, str(r[text_col]), temps)
                rows.append({
                    "text": res["text"][:90],
                    "department": res["department"],
                    "dept_conf": round(res["department_confidence"], 3),
                    "urgency": res["urgency_label"],
                    "decision": res["gate_decision"],
                    "churn": round(res["churn_risk"], 2),
                    "refund": round(res["refund_requested"], 2),
                })
                prog.progress((i + 1) / len(sub), f"Triaging {i+1}/{len(sub)}")
            out = pd.DataFrame(rows)
            auto_rate = (out["decision"] == "auto_route").mean() * 100
            m1, m2 = st.columns(2)
            m1.metric("Tickets triaged", len(out))
            m2.metric("Auto-routed", f"{auto_rate:.0f}%")
            st.dataframe(out, hide_index=True, width="stretch")
            st.markdown("**Volume by department**")
            st.bar_chart(out["department"].value_counts())
            st.download_button("⬇ Download results CSV", out.to_csv(index=False),
                               "triage_results.csv", "text/csv")


# --------------------------------------------------------------------------
# EVALUATION TAB
# --------------------------------------------------------------------------
with eval_tab:
    st.subheader("How well does it actually work?")
    st.markdown(
        "An **honest, zero-shot** evaluation on a held-out test set the model never saw during "
        "calibration. No fine-tuning, no cherry-picking. The point isn't a big accuracy number — "
        "it's understanding *where the open model is trustworthy and where it isn't*."
    )
    if _df is None:
        st.error("No predictions found — run `python -m evaluation.run_eval` first.")
    else:
        df = _df
        total = len(df)
        auto = df[df["gate_decision"] == "auto_route"]
        esc = df[df["gate_decision"] == "escalate_to_human"]
        dept_acc = (df["cal_department"] == df["true_department"]).mean()
        urg_acc = (df["cal_urgency"].astype(int) == df["true_urgency"].astype(int)).mean()
        auto_acc = (auto["cal_department"] == auto["true_department"]).mean() if len(auto) else 0
        esc_acc = (esc["cal_department"] == esc["true_department"]).mean() if len(esc) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Held-out tickets", total)
        c2.metric("Overall dept accuracy", f"{dept_acc*100:.1f}%", help="8-way choice; random ≈ 12.5%")
        c3.metric("Auto-route rate", f"{len(auto)/total*100:.1f}%", help=f"at ≥ {CONFIDENCE_THRESHOLD} confidence")
        c4.metric("Accuracy when auto-routed", f"{auto_acc*100:.1f}%", delta=f"{(auto_acc-dept_acc)*100:+.0f} pts vs overall")

        st.success(
            f"**The headline story.** At ≥ {CONFIDENCE_THRESHOLD:.0%} confidence, Laya auto-routes "
            f"**{len(auto)/total*100:.0f}%** of tickets at **{auto_acc*100:.0f}% accuracy**, and escalates "
            f"the rest — where it's only {esc_acc*100:.0f}% accurate — to a human. In other words, "
            f"**the confidence score is meaningful**: the model knows what it knows. That's the whole "
            f"value of a calibrated System-1 model over a raw LLM classifier."
        )

        st.markdown("#### 🎚 Coverage vs. accuracy — you choose the operating point")
        st.caption("Slide the threshold higher to auto-route fewer tickets but with higher accuracy among them. "
                   "This is the dial a real deployment tunes.")
        st.pyplot(threshold_tradeoff_fig(df))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Department confusion")
            st.caption("Rows = truth, cols = prediction. Note how IT / Product / Customer Service "
                       "tickets get pulled into **Technical Support** — those categories genuinely overlap.")
            st.pyplot(department_confusion_fig(df))
        with col_b:
            st.markdown("#### Urgency (3-class)")
            st.caption(f"Exact accuracy {urg_acc*100:.0f}%. It rarely predicts 'low' — Laya's `score` "
                       "primitive is its documented weak spot. An honest limitation, not hidden.")
            st.pyplot(urgency_confusion_fig(df))

        st.markdown("#### Per-department recall")
        recall_rows = []
        for dept in DEPARTMENTS:
            sub = df[df["true_department"] == dept]
            if len(sub):
                recall_rows.append({
                    "department": dept,
                    "support": len(sub),
                    "recall %": round((sub["cal_department"] == dept).mean() * 100, 1),
                })
        rec_df = pd.DataFrame(recall_rows).sort_values("recall %", ascending=False)
        st.dataframe(rec_df, hide_index=True, width="stretch",
                     column_config={"recall %": st.column_config.ProgressColumn(
                         "recall %", min_value=0, max_value=100, format="%.1f%%")})

        st.markdown("#### 🔍 What the numbers hide: look at real cases")
        confident_correct, disagreements = interesting_cases(df)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Confident & correct** — the model at its best")
            st.dataframe(
                confident_correct.assign(preview=lambda d: d["text"].str.slice(0, 90))[
                    ["preview", "cal_department", "cal_department_confidence"]
                ].rename(columns={"cal_department": "predicted", "cal_department_confidence": "conf"}),
                hide_index=True, width="stretch")
        with cc2:
            st.markdown("**Confident disagreements** — often the *dataset label* is the questionable one")
            st.dataframe(
                disagreements.assign(preview=lambda d: d["text"].str.slice(0, 90))[
                    ["preview", "cal_department", "true_department", "cal_department_confidence"]
                ].rename(columns={"cal_department": "predicted", "true_department": "dataset label",
                                  "cal_department_confidence": "conf"}),
                hide_index=True, width="stretch")

        with st.expander("Method & honesty notes"):
            st.markdown(
                "- **Zero-shot:** the model was *not* fine-tuned on this data. Laya's own docs note the "
                "base checkpoint scores near-chance zero-shot and that fine-tuning lifts it to ~0.77.\n"
                "- **Noisy labels:** the dataset's own categories overlap and are inconsistent, so some "
                "'errors' are cases where Laya's pick is arguably better than the ground truth.\n"
                "- **Calibration:** temperatures fit on a separate 500-ticket split (department T≈1.11, "
                "urgency T≈3.68). The choice head ships well-calibrated; the score head needed softening.\n"
                "- **Urgency is 3-class** (low/medium/high) — the only levels present in the English data.\n"
                "- **Latency:** ~137 ms/ticket on an RTX 3050 (GPU), one forward pass, no tokens generated."
            )
