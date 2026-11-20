"""evaluate.py - runs BOTH models on the SAME inputs and logs everything to MLflow.

Usage:
    python evaluate.py                 # full run (200 samples)
    N_SAMPLES=10 python evaluate.py    # quick debug run  (Windows PowerShell: $env:N_SAMPLES=10; python evaluate.py)
"""
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

from hf_wrapper import HFSentimentModel, normalize_label

# ----------------------------------------------------------------- settings
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "hf-sentiment-champion-challenger"
TASK = "sentiment"
DATASET_NAME = "glue/sst2 (validation split)"
SEED = 42
N_SAMPLES = int(os.environ.get("N_SAMPLES", 200))
MAX_LENGTH = 128
BATCH_SIZE = 1  # batch size 1 => clean per-example latency
USN = os.environ.get("USN", "YOUR_USN")  # set your USN: $env:USN="1MS23..."

MODEL_A = dict(
    key="A", role="champion",
    run_name="A_sentiment_distilbert-sst2",
    short="distilbert-sst2",
    hf_model_id="distilbert-base-uncased-finetuned-sst-2-english",
)
MODEL_B = dict(
    key="B", role="challenger",
    run_name="B_sentiment_roberta-large",
    short="roberta-large",
    hf_model_id="siebert/sentiment-roberta-large-english",
)

# ---- Champion rule (MUST match README.md; written BEFORE the final run) ----
PRIMARY_METRIC = "macro_f1"          # higher is better
MARGIN = 0.01                        # Challenger must beat Champion by >= 1 pp
MAX_LATENCY_RATIO = 10.0             # guardrail: reject Challenger if > 10x slower
TIE_BREAKER = "if |delta| < margin, prefer the faster model (lower avg_latency_ms)"

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "macro_f1"]  # higher = better
LOWER_IS_BETTER = ["avg_latency_ms", "total_inference_s"]


# ------------------------------------------------------------------- data

def prepare_eval_set() -> pd.DataFrame:
    """Create (or reuse) the shared evaluation set and save it to data/eval_inputs.csv."""
    path = DATA_DIR / "eval_inputs.csv"

    if path.exists():
        df = pd.read_csv(path)
        if len(df) == N_SAMPLES:
            print(f"Reusing existing {path} ({len(df)} rows)")
            return df

    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(ds), size=N_SAMPLES, replace=False)
    df = pd.DataFrame(
        {
            "sample_id": idx,
            "text": [ds[int(i)]["sentence"].strip() for i in idx],
            "reference": [int(ds[int(i)]["label"]) for i in idx],
        }
    )
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")
    return df


# ------------------------------------------------------------- inference
def run_model(hf_model_id: str, texts: list[str]):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("sentiment-analysis", model=hf_model_id, device=device)
    for t in texts[:3]:  # warm-up (not timed)
        pipe(t, truncation=True, max_length=MAX_LENGTH)

    preds, names, scores, times = [], [], [], []
    for t in texts:
        start = time.perf_counter()
        out = pipe(t, truncation=True, max_length=MAX_LENGTH)[0]
        times.append(time.perf_counter() - start)
        preds.append(normalize_label(out["label"]))
        names.append(out["label"])
        scores.append(float(out["score"]))
    return preds, names, scores, times, ("cuda" if device == 0 else "cpu")


def compute_metrics(y_true, y_pred, times) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),  # positive class
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "avg_latency_ms": float(np.mean(times) * 1000),
        "total_inference_s": float(np.sum(times)),
    }


# --------------------------------------------------------- model logging
def log_hf_model(hf_model_id: str):
    """Log the wrapper as an MLflow pyfunc model (works on MLflow 2.x and 3.x)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "config.json"
        cfg_path.write_text(
            json.dumps(
                {"hf_model_id": hf_model_id, "task": TASK,
                 "max_length": MAX_LENGTH, "batch_size": 8}
            )
        )
        kwargs = dict(
            python_model=HFSentimentModel(),
            artifacts={"config": str(cfg_path)},
            code_paths=["hf_wrapper.py"],
            input_example=pd.DataFrame({"text": ["A great movie."]}),
            pip_requirements=["mlflow", "transformers", "torch", "pandas"],
        )
        try:
            mlflow.pyfunc.log_model(name="model", **kwargs)  # MLflow 3.x
        except TypeError:
            mlflow.pyfunc.log_model(artifact_path="model", **kwargs)  # MLflow 2.x


# ------------------------------------------------------------------ plots
def confusion_fig(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def comparison_figs(ma: dict, mb: dict):
    x = np.arange(len(METRIC_KEYS))
    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [ma[k] for k in METRIC_KEYS], 0.4, label=f"A {MODEL_A['short']}")
    ax.bar(x + 0.2, [mb[k] for k in METRIC_KEYS], 0.4, label=f"B {MODEL_B['short']}")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Champion (A) vs Challenger (B) - quality metrics (higher is better)")
    ax.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(["A", "B"], [ma["avg_latency_ms"], mb["avg_latency_ms"]],
            color=["tab:blue", "tab:orange"])
    ax2.set_ylabel("avg latency per example (ms)")
    ax2.set_title("Latency (lower is better)")
    fig2.tight_layout()
    return fig1, fig2


# --------------------------------------------------------------- decision
def decide(ma: dict, mb: dict):
    delta = mb[PRIMARY_METRIC] - ma[PRIMARY_METRIC]
    ratio = mb["avg_latency_ms"] / ma["avg_latency_ms"]
    if ratio > MAX_LATENCY_RATIO:
        return "A", delta, ratio, f"guardrail: B is {ratio:.1f}x slower than A"
    if delta >= MARGIN:
        return "B", delta, ratio, f"B beats A by {delta:.4f} >= margin {MARGIN}"
    if delta <= -MARGIN:
        return "A", delta, ratio, f"A beats B by {-delta:.4f} >= margin {MARGIN}"
    faster = "A" if ma["avg_latency_ms"] <= mb["avg_latency_ms"] else "B"
    return faster, delta, ratio, (
        f"difference {delta:+.4f} within margin {MARGIN}; tie-breaker -> faster model ({faster})"
    )


# ------------------------------------------------------------------- main
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = prepare_eval_set()
    texts = df["text"].tolist()
    y_true = df["reference"].tolist()
    lib_versions = (
        f"transformers={transformers.__version__}, torch={torch.__version__}, "
        f"mlflow={mlflow.__version__}, python={platform.python_version()}"
    )
    common_tags = {
        "student_usn": USN,
        "library_versions": lib_versions,
        "dataset_name": DATASET_NAME,
    }

    results = {}  # key -> dict(metrics, preds_df, run_id)

    with mlflow.start_run(run_name=f"compare_{TASK}") as parent:
        mlflow.set_tags(common_tags)

        for m in (MODEL_A, MODEL_B):
            print(f"\n=== {m['run_name']} : {m['hf_model_id']} ===")
            preds, names, scores, times, device = run_model(m["hf_model_id"], texts)
            metrics = compute_metrics(y_true, preds, times)
            pred_df = df.assign(
                prediction=preds, label_name=names,
                confidence=np.round(scores, 4),
                correct=[int(a == b) for a, b in zip(y_true, preds)],
            )

            with mlflow.start_run(run_name=m["run_name"], nested=True) as child:
                mlflow.log_params({
                    "hf_model_id": m["hf_model_id"], "task": TASK, "role": m["role"],
                    "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
                    "eval_samples": len(df), "seed": SEED, "device": device,
                })
                mlflow.set_tags({**common_tags, "role": m["role"]})
                mlflow.log_metrics(metrics)

                # artifacts
                pred_path = OUT_DIR / f"predictions_{m['short']}.csv"
                pred_df.to_csv(pred_path, index=False)
                metrics_path = OUT_DIR / f"metrics_summary_{m['short']}.json"
                metrics_path.write_text(json.dumps(
                    {"model": m["hf_model_id"], "metrics": metrics}, indent=2))
                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(metrics_path), artifact_path="")
                mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))

                fig = confusion_fig(y_true, preds, f"{m['short']} - confusion matrix")
                mlflow.log_figure(fig, "plots/confusion_matrix.png")
                fig.savefig(OUT_DIR / f"confusion_matrix_{m['short']}.png", dpi=150)
                plt.close(fig)

                log_hf_model(m["hf_model_id"])
                results[m["key"]] = dict(metrics=metrics, pred_df=pred_df,
                                         run_id=child.info.run_id)
            print({k: round(v, 4) for k, v in metrics.items()})

        # ---------------- summary / decision (parent run)
        ma, mb = results["A"]["metrics"], results["B"]["metrics"]
        winner, delta, ratio, reason = decide(ma, mb)
        winner_model = MODEL_A if winner == "A" else MODEL_B

        mlflow.log_params({
            "primary_metric": PRIMARY_METRIC, "margin": MARGIN,
            "max_latency_ratio": MAX_LATENCY_RATIO, "tie_breaker": TIE_BREAKER,
            "direction_quality_metrics": "higher is better",
            "direction_latency": "lower is better",
            "task": TASK, "eval_samples": len(df), "seed": SEED,
            "model_A": MODEL_A["hf_model_id"], "model_B": MODEL_B["hf_model_id"],
        })
        mlflow.set_tags({
            "winner": winner, "winner_hf_model_id": winner_model["hf_model_id"],
            "decision_reason": reason,
            "run_id_A": results["A"]["run_id"], "run_id_B": results["B"]["run_id"],
        })
        mlflow.log_metric("delta_primary_metric", delta)  # challenger - champion
        mlflow.log_metric("latency_ratio_B_over_A", ratio)
        for k, v in ma.items():
            mlflow.log_metric(f"A_{k}", v)
        for k, v in mb.items():
            mlflow.log_metric(f"B_{k}", v)

        # artifacts on the summary run
        summary = {
            "primary_metric": PRIMARY_METRIC, "margin": MARGIN,
            "A": {"hf_model_id": MODEL_A["hf_model_id"], **ma},
            "B": {"hf_model_id": MODEL_B["hf_model_id"], **mb},
            "delta_primary_metric": delta, "latency_ratio_B_over_A": ratio,
            "winner": winner, "reason": reason,
        }
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(OUT_DIR / "metrics_summary.json"))
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))

        fig1, fig2 = comparison_figs(ma, mb)
        mlflow.log_figure(fig1, "plots/metrics_comparison.png")
        mlflow.log_figure(fig2, "plots/latency_comparison.png")
        fig1.savefig(OUT_DIR / "metrics_comparison.png", dpi=150)
        fig2.savefig(OUT_DIR / "latency_comparison.png", dpi=150)
        plt.close("all")

        # qualitative examples: where the models disagree
        a_df, b_df = results["A"]["pred_df"], results["B"]["pred_df"]
        dis = pd.DataFrame({
            "text": a_df["text"], "reference": a_df["reference"],
            "pred_A": a_df["prediction"], "pred_B": b_df["prediction"],
            "conf_A": a_df["confidence"], "conf_B": b_df["confidence"],
        })
        dis = dis[dis["pred_A"] != dis["pred_B"]].head(5).copy()
        dis["comment"] = ""  # <-- write your own short comment for each row
        dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
        mlflow.log_artifact(str(OUT_DIR / "disagreements.csv"))

        if Path("README.md").exists():
            mlflow.log_artifact("README.md")
        mlflow.log_artifact("requirements.txt")

        print("\n================ DECISION ================")
        print(f"{PRIMARY_METRIC}: A={ma[PRIMARY_METRIC]:.4f}  B={mb[PRIMARY_METRIC]:.4f}  "
              f"delta(B-A)={delta:+.4f}")
        print(f"latency: A={ma['avg_latency_ms']:.1f} ms  B={mb['avg_latency_ms']:.1f} ms "
              f"(ratio {ratio:.1f}x)")
        print(f"WINNER: Model {winner} ({winner_model['hf_model_id']})")
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
