"""register_champion.py - registers both logged models and sets the aliases.

Usage:
    python register_champion.py          # register A and B, alias 'champion' -> winner
    python register_champion.py --flip   # swap the two aliases (stretch goal)
"""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "hf-sentiment-champion-challenger"
REGISTERED_MODEL = "sentiment-model"
TASK = "sentiment"
RUN_A = "A_sentiment_distilbert-sst2"
RUN_B = "B_sentiment_roberta-large"
RUN_PARENT = f"compare_{TASK}"


def latest_run(client, exp_id, name):
    runs = client.search_runs(
        [exp_id], filter_string=f"tags.mlflow.runName = '{name}'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )
    if not runs:
        raise SystemExit(f"Run '{name}' not found. Run evaluate.py first.")
    return runs[0]


def get_or_register(client, run):
    """Register runs:/<id>/model once; reuse the version if already registered."""
    for mv in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        if mv.run_id == run.info.run_id:
            return mv
    return mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip", action="store_true", help="swap champion/challenger aliases")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()

    if args.flip:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
        chall = client.get_model_version_by_alias(REGISTERED_MODEL, "challenger")
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", chall.version)
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", champ.version)
        print(f"FLIPPED: champion -> v{chall.version}, challenger -> v{champ.version}")
        return

    exp = client.get_experiment_by_name(EXPERIMENT)
    parent = latest_run(client, exp.experiment_id, RUN_PARENT)
    run_a = latest_run(client, exp.experiment_id, RUN_A)
    run_b = latest_run(client, exp.experiment_id, RUN_B)
    winner = parent.data.tags["winner"]  # decided by evaluate.py using the README rule
    print(f"Decision from summary run: winner = Model {winner}")

    mv_a = get_or_register(client, run_a)
    mv_b = get_or_register(client, run_b)
    for mv, run in ((mv_a, run_a), (mv_b, run_b)):
        client.set_model_version_tag(
            REGISTERED_MODEL, mv.version, "hf_model_id", run.data.params["hf_model_id"])

    win_mv, lose_mv = (mv_a, mv_b) if winner == "A" else (mv_b, mv_a)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", win_mv.version)
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", lose_mv.version)

    info = {
        "registered_model": REGISTERED_MODEL,
        "champion_version": win_mv.version, "challenger_version": lose_mv.version,
        "version_A": mv_a.version, "version_B": mv_b.version, "winner": winner,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/registry_summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
