# Champion–Challenger Sentiment Classification with Hugging Face + MLflow

**Student USN:** `1MS25CI106` | **Course:** Skill Lab (MLOps with MLflow), RIT

## 1. Task and Models

- **Task:** Binary sentiment classification (positive / negative).
- **Dataset:** GLUE SST-2 validation split, 200 samples selected using `seed=42` and saved in `data/eval_inputs.csv`.
- **Model A (Previous Champion):** `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters).
- **Model B (New Champion):** `siebert/sentiment-roberta-large-english` (~355M parameters).
- Both models were evaluated on the same inputs in the same order.
- Raw model labels (`POSITIVE` / `NEGATIVE`) were mapped to 1 / 0.

## 2. How to Run

### Step 1: Create and activate the environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Start the MLflow server

Run this command in the first terminal and keep it running:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### Step 3: Run the evaluation

Open another terminal:

```powershell
$env:USN="1MS25CI106"
python evaluate.py
```

### Step 4: Register the winning model

```powershell
python register_champion.py
```

### Step 5: Generate predictions

```powershell
python predict_from_mlflow.py
python predict_from_mlflow.py --both
```

### MLflow Dashboard

http://127.0.0.1:5000

## 3. Champion Selection Rule

The following rules were defined before the final evaluation:

| Part | Rule |
|---|---|
| Primary metric | Macro-F1 on the 200-sample evaluation set |
| Minimum improvement | Challenger must improve macro-F1 by at least 0.01 |
| Tie-breaker | If the difference is below 0.01, the faster model wins |
| Latency guardrail | Challenger is rejected if it is more than 10 times slower |
| Metric direction | Higher accuracy, precision, recall, F1 and macro-F1 are better; lower latency is better |

## 4. Evaluation Results

Final results from the 200-sample evaluation:

| Metric | A: DistilBERT | B: RoBERTa Large | Difference (B − A) |
|---|---:|---:|---:|
| Accuracy | 0.9350 | 0.9450 | +0.0100 |
| Precision | 0.9286 | 0.9375 | +0.0089 |
| Recall | 0.9541 | 0.9633 | +0.0092 |
| F1 | 0.9412 | 0.9502 | +0.0090 |
| **Macro-F1** | **0.9343** | **0.9444** | **+0.0101** |
| Average latency (ms) | 26.49 | 193.36 | +166.87 |
| Total inference (s) | 5.30 | 38.67 | +33.37 |

The evaluation results are saved in `outputs/metrics_summary.json`.

Plots and comparison artifacts are saved in `outputs/`. Model comparison, confusion matrices and latency visualizations are logged in MLflow.

## 5. Final Champion–Challenger Decision

The Challenger model, RoBERTa Large, outperformed the previous Champion, DistilBERT, on the 200-sample SST-2 evaluation set. Its macro-F1 score was 0.9444 compared with DistilBERT's 0.9343, giving an improvement of 0.0101 (approximately 1.01 percentage points). This exceeded the predefined minimum improvement margin of 0.01, so the Challenger was selected as the new Champion.

RoBERTa also achieved higher accuracy (0.945 versus 0.935), precision, recall and F1 score. However, this improvement comes with a speed trade-off. Its average inference latency was 193.36 ms, compared with 26.49 ms for DistilBERT, making it approximately 7.3 times slower. This remains within the predefined 10x latency guardrail.

The improvement is relatively small, and because the evaluation contains only 200 samples, it should not be considered conclusive evidence of general superiority. Overall, RoBERTa was selected because it crossed the required macro-F1 improvement threshold without violating the latency limit. DistilBERT remains a useful alternative when faster inference is more important than the small improvement in classification performance.

## 6. Disagreement Analysis

The following five examples were identified where the two models predicted different sentiments:

1. **Pleasant romance:** DistilBERT correctly predicted positive sentiment. RoBERTa predicted negative, possibly due to the phrase "maddeningly predictable."

2. **Fun and entertaining movie:** DistilBERT correctly predicted positive sentiment. RoBERTa may have struggled with the informal wording and short descriptive phrases.

3. **Bit disappointing:** DistilBERT correctly predicted negative sentiment. RoBERTa may have focused on the positive word "decent" and overlooked the negative context.

4. **Oh come on:** RoBERTa correctly predicted negative sentiment. DistilBERT predicted positive, possibly because the short phrase lacks detailed context.

5. **Cannes movie review:** DistilBERT correctly predicted negative sentiment. RoBERTa may have focused on the positive expression "always dreamed" instead of the overall negative meaning.

These examples show that a higher overall evaluation score does not mean a model will classify every individual sentence correctly.

The original disagreement records are available in `outputs/disagreements.csv`.

## 7. Model Registry

The registered MLflow model is named `sentiment-model`.

| Alias | Version | Model |
|---|---:|---|
| Champion | 2 | `siebert/sentiment-roberta-large-english` |
| Challenger | 1 | `distilbert-base-uncased-finetuned-sst-2-english` |

The prediction script loads models using their MLflow aliases, allowing the active Champion and Challenger to be tested without manually specifying model versions.

## 8. Limitations and Declaration

- Only 200 evaluation samples were used, so small performance differences may be due to sampling variation.
- SST-2 contains short movie-review sentences, and results may not generalize to other domains.
- The DistilBERT model was fine-tuned on SST-2 training data. SiEBERT was trained using multiple datasets, which may include overlapping sentiment data.
- Latency was measured on the local machine and may vary depending on hardware and system load.
- Inference was performed with batch size 1.

**Tools used:** Hugging Face Transformers, MLflow, scikit-learn, pandas and matplotlib.

An AI assistant (ChatGPT) helped scaffold the project and README. The implementation and results were reviewed as part of the project work.