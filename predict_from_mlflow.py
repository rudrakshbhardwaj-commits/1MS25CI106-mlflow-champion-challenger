"""predict_from_mlflow.py - loads a model by registry ALIAS and predicts on new data.

NOTE: this script deliberately does NOT import transformers / call a Hugging Face
pipeline. Everything goes through MLflow.

Usage:
    python predict_from_mlflow.py                  # champion
    python predict_from_mlflow.py --both           # champion + challenger
    python predict_from_mlflow.py --alias challenger
"""
import argparse
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL = "sentiment-model"
LABELS = {1: "positive", 0: "negative"}


def predict_with_alias(client, alias, new_df):
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    run = client.get_run(mv.run_id)
    hf_id = mv.tags.get("hf_model_id") or run.data.params.get("hf_model_id", "unknown")
    print(f"\n>>> Alias '@{alias}' -> {REGISTERED_MODEL} version {mv.version} | HF model: {hf_id}")

    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{alias}")
    out = model.predict(new_df[["text"]])

    res = new_df.copy().reset_index(drop=True)
    res["alias"] = alias
    res["model_version"] = mv.version
    res["hf_model_id"] = hf_id
    res["prediction"] = out["prediction"].values
    res["sentiment"] = res["prediction"].map(LABELS)
    res["confidence"] = out["confidence"].values
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--both", action="store_true", help="run champion and challenger")
    ap.add_argument("--input", default="data/new_data.csv")
    ap.add_argument("--output", default="outputs/new_data_predictions.csv")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()

    new_df = pd.read_csv(args.input)
    aliases = ["champion", "challenger"] if args.both else [args.alias]
    frames = [predict_with_alias(client, a, new_df) for a in aliases]
    result = pd.concat(frames, ignore_index=True)

    show = result.copy()
    show["text"] = show["text"].str.slice(0, 55)
    cols = ["alias", "model_version", "text", "sentiment", "confidence"]
    if "expected_label" in show.columns:
        show["expected"] = show["expected_label"].map(LABELS)
        cols.insert(4, "expected")
    print("\n" + show[cols].to_string(index=False))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
