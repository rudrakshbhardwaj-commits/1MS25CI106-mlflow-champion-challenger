# Champion–Challenger Sentiment Classification with Hugging Face + MLflow

**Student USN:** `<YOUR_USN>`  |  **Course:** Skill Lab (MLOps with MLflow), RIT

## 1. Task and models

- **Task:** binary sentiment classification (positive / negative)
- **Dataset:** GLUE SST-2, validation split, **200 samples** drawn with `seed=42` (saved in `data/eval_inputs.csv`)
- **Model A (current Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M)
- **Model B (Challenger):** `siebert/sentiment-roberta-large-english` (~355M)
- Both models run on the same inputs, in the same order. Raw labels (`POSITIVE`/`NEGATIVE`) are mapped to 1/0.

## 2. How to run

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. terminal 1 - start MLflow server (keep running)
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 3. terminal 2 - run everything (set your USN first)
$env:USN="<YOUR_USN>"             # PowerShell  (bash: export USN=...)
python evaluate.py                # runs both models, logs to MLflow, makes the decision
python register_champion.py       # registers models, sets @champion / @challenger
python predict_from_mlflow.py     # loads @champion, predicts on data/new_data.csv
python predict_from_mlflow.py --both

# stretch goal: flip aliases and re-run WITHOUT editing the script
python register_champion.py --flip
python predict_from_mlflow.py
```

MLflow UI: http://127.0.0.1:5000

## 3. Champion rule (written BEFORE the final run, never changed afterwards)

| Part | Rule |
| --- | --- |
| Primary metric | **macro-F1** on the 200-sample evaluation set (**higher is better**) |
| Minimum margin | Challenger must beat Champion by **>= 0.01 (1 percentage point)**. With only 200 samples, 1 pp is just 2 examples, so anything smaller is treated as noise. |
| Tie-breaker | If \|delta\| < margin, the **faster** model (lower `avg_latency_ms`) wins. |
| Guardrail | Challenger is rejected if its latency is **> 10x** the Champion's. |
| Direction | accuracy / precision / recall / F1 / macro-F1: higher is better. `avg_latency_ms`, `total_inference_s`: lower is better. |

## 4. Results

> Fill this in from `outputs/metrics_summary.json` after running.

| Metric | A: distilbert-sst2 | B: roberta-large | Difference (B − A) |
| --- | --- | --- | --- |
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1 | | | |
| **Macro-F1 (primary)** | | | |
| Avg latency (ms) | | | |
| Total inference (s) | | | |

Plots are in `outputs/` (confusion matrices, metric comparison, latency comparison).
Disagreement examples with my comments are in `outputs/disagreements.csv`.

## 5. Decision (150–250 words)

> Write this yourself after seeing the results. Answer: Does the Challenger improve on the
> Champion? By how much? Is the difference meaningful for 200 samples? What are the
> trade-offs (speed, size, errors)? Remember: a Challenger that does not win is a fine result.

## 6. Limitations and declaration

- Only 200 evaluation samples; a ~1 pp difference is about 2 examples and may be noise.
- SST-2 is short movie-review text; results may not generalise to other domains.
- The DistilBERT SST-2 model was fine-tuned on SST-2 *train*; SiEBERT was trained on a mix of
  datasets that may include SST-2 data, so scores could be optimistic.
- Latency measured on my machine (CPU/GPU noted in MLflow param `device`), batch size 1.
- **Tools used:** Hugging Face Transformers, MLflow, scikit-learn, matplotlib.
  An AI assistant (Claude) helped me scaffold the code and the README; I have read and can explain every line.
