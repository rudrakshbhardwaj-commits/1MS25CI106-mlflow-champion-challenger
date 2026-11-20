"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (Lab 9 style).

Input : DataFrame with a 'text' column (or a list / Series of strings)
Output: DataFrame with text, prediction (1=positive, 0=negative), label_name,
        confidence, hf_model_id
"""
import json

import mlflow.pyfunc
import pandas as pd


def normalize_label(label: str) -> int:
    """Map any model's raw label to 1 (positive) / 0 (negative).

    Handles POSITIVE/NEGATIVE (distilbert, siebert) and LABEL_0/LABEL_1 (textattack).
    """
    l = str(label).strip().lower()
    if l in ("positive", "pos", "label_1", "1"):
        return 1
    if l in ("negative", "neg", "label_0", "0"):
        return 0
    raise ValueError(f"Unknown label from model: {label!r} (add it to normalize_label)")


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Imported here so that scripts which only *load* the MLflow model
        # never import Hugging Face in their own code.
        import torch
        from transformers import pipeline

        with open(context.artifacts["config"]) as f:
            self.cfg = json.load(f)
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "sentiment-analysis", model=self.cfg["hf_model_id"], device=device
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in list(model_input)]

        raw = self.pipe(
            texts,
            truncation=True,
            max_length=int(self.cfg.get("max_length", 128)),
            batch_size=int(self.cfg.get("batch_size", 8)),
        )
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [normalize_label(r["label"]) for r in raw],
                "label_name": [r["label"] for r in raw],
                "confidence": [round(float(r["score"]), 4) for r in raw],
                "hf_model_id": self.cfg["hf_model_id"],
            }
        )
