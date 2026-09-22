# laya-support-ticket-triage

A local, GPU-accelerated **customer-support ticket triage system** built on **Laya**, the open-source *System-1 decision engine*. It reads a support ticket and returns typed, calibrated decisions — which department should handle it, how urgent it is, whether the customer is a churn or refund risk — in a **single forward pass**, with no text generation to parse and nothing to hallucinate. The project then **benchmarks Laya honestly** against a held-out set of real tickets and ships an interactive dashboard to explore the results.


---

## Background: Jev and Laya

**Jev** is a proprietary "System One" model from **TypeSafe AI**, launched September 2026 by Diogo Almeida (ex-OpenAI, co-author of RLHF/InstructGPT). Unlike a chat LLM, it doesn't generate text — it takes unstructured input plus predefined typed questions (choice / score / boolean) and returns machine-readable probabilities with calibrated confidence, in tens to hundreds of milliseconds. It's a closed, managed API aimed at fast, narrow decision tasks.

- Wikipedia: https://en.wikipedia.org/wiki/Jev_(AI_model)
- TypeSafe launch write-up (via community coverage): https://www.truefoundry.com/blog/typesafe-ai-jev

**Laya** is the open-source counterpart from **Convai Innovations** (Apache-2.0). Same idea — a non-autoregressive System-1 decision engine that answers typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket, JSON) in one forward pass (~33 ms on a T4) — but the weights are open and you can self-host. It ships English, multilingual, and fine-tuned checkpoints plus a built-in router.

- Hugging Face model: https://huggingface.co/convaiinnovations/laya
- PyPI: https://pypi.org/project/laya/
- GitHub: https://github.com/NandhaKishorM/laya

---

## What we built

Using Laya as the decision engine, this project builds and honestly evaluates a full triage pipeline:

- **Data pipeline** — downloads a real support-ticket dataset, cleans it, and maps it to an 8-department label space and a 3-level urgency scale, with a seeded 3-way split (calibrate / dev / test) so the evaluation stays honest.
- **Decision pipeline** — runs each ticket through Laya's English checkpoint with our typed questions, plus **temperature calibration** so the confidence scores are meaningful.
- **Confidence gating** — auto-route tickets the model is confident about, escalate the rest to a human.
- **Honest evaluation** — accuracy, confusion matrices, per-department recall, the auto-route-rate vs accuracy tradeoff, and latency, all on the untouched test set.
- **SQLite logging** — every triaged ticket is recorded and queryable.
- **Interactive dashboard** — a Streamlit "Live Demo" with Live, Batch, and Evaluation tabs.

**Headline result (939 held-out tickets, zero-shot, no fine-tuning):** at ≥ 0.85 confidence, Laya **auto-routes ~7% of tickets at 97% accuracy** and correctly escalates the rest — the confidence score is genuinely meaningful. Overall department accuracy is 39% on an 8-way choice (~3× random), strong on distinct classes and weak where categories overlap. Urgency is the model's documented weak spot, shown honestly. Runs at ~137 ms/ticket on an RTX 3050.

---

## Tech Stack

- **Laya** (`laya`) — the open-source System-1 decision engine (weights from Hugging Face)
- **PyTorch** — runtime (CUDA build for GPU)
- **Hugging Face `datasets`** — dataset download
- **pandas / pyarrow** — data wrangling + parquet caching
- **scikit-learn / NumPy** — evaluation metrics
- **matplotlib** — confusion matrices and the tradeoff curve
- **Streamlit** — the interactive dashboard
- **SQLite** (stdlib) — logging store
- **Python 3.10+**

---

## Project Structure

```
laya-support-ticket-triage/
├── config.py                 # paths, label spaces, VRAM-safe defaults, device auto-detect
├── requirements.txt
├── data/
│   ├── loader.py             # download -> clean -> map labels -> 3-way stratified split
│   ├── labels.py             # queue -> department, priority -> urgency mapping
│   └── check_labels.py       # runnable check (no GPU)
├── laya_pipeline/
│   ├── questions.py          # the typed questions asked of Laya
│   ├── engine.py             # load router, predict one/batch, run a split
│   ├── gating.py             # confidence gating + temperature calibration
│   └── check_gating.py       # runnable check (no GPU)
├── evaluation/
│   ├── metrics.py            # department / urgency / gating / latency metrics
│   ├── report.py             # markdown report builder
│   ├── run_eval.py           # full test-set evaluation entrypoint
│   └── check_metrics.py      # runnable check (no GPU)
├── storage/
│   ├── db.py                 # SQLite logging (log, backfill, summary, query)
│   └── check_db.py           # runnable check (no GPU)
├── dashboard/
│   ├── app.py                # Streamlit "Live Demo" (Live / Batch / Evaluation)
│   ├── helpers.py            # cached router + single-ticket triage
│   ├── charts.py             # matplotlib figures for the Evaluation tab
│   └── examples.py           # prebuilt example tickets for the Live tab
└── reports/                  # generated evaluation_report.md + predictions.parquet
```

---

## Installation

```bash

# 1. Clone the repository: 
git clone https://github.com/IshaanLabs/laya-support-ticket-triage.git

# 2. Create and activate a virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU) Install a PyTorch build matching your driver.
#    Example for CUDA 12.1-compatible drivers:
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
```

Laya's model weights download automatically from Hugging Face on first use and are cached locally. CPU works out of the box (slower); the pipeline auto-detects the device.

---

## Configuration

All tunables live in `config.py`:

- `HF_DATASET`, `HF_DATA_FILE` — dataset source
- `DEPARTMENTS` — the 8 support departments (the choice label space)
- `PRIORITY_TO_SCORE`, `URGENCY_RUBRIC` — the 3-level urgency scale
- `SPLIT_SIZES`, `SPLIT_SEED` — the calibrate / dev / test split
- `LAYA_CHECKPOINTS` — which checkpoints to preload (English only, to fit 4 GB VRAM)
- `BATCH_SIZE` — lower to 4 if you hit CUDA out-of-memory
- `CONFIDENCE_THRESHOLD` — the auto-route vs escalate cutoff (default 0.85)

Set `LAYA_FORCE_CPU=1` to force CPU regardless of GPU availability.

---

## Usage

Run the pipeline stage by stage:

```bash
python -m data.loader                 # 1. build the dataset splits
python -m laya_pipeline.gating        # 2. fit calibration temperatures
python -m evaluation.run_eval         # 3. full evaluation -> reports/
python -m storage.db --backfill       # 4. log results to SQLite
streamlit run dashboard/app.py        # 5. launch the dashboard
```

Query the logged tickets:

```bash
python -m storage.db --summary
python -m storage.db --show escalate_to_human --limit 10
```

Run the fast, GPU-free logic checks:

```bash
python -m data.check_labels
python -m laya_pipeline.check_gating
python -m evaluation.check_metrics
python -m storage.check_db
```

If Streamlit's file-watcher prints an unrelated `torchvision` probe error, run it with the watcher off:

```bash
streamlit run dashboard/app.py --server.fileWatcherType none
```

---


## Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License - see the MIT License file for details.

Laya itself is licensed under Apache-2.0 by Convai Innovations. Dataset: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets).
