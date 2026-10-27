"""
Train the deployable model on every month (dev + test) and save it with
reference score distributions for percentile reporting.

The deployed score is blend_pre_eq, the model declared primary before the
held-out test: the mean of two within-month percentiles among all at-risk
packages, one from the popularity prior (dependents) and one from a GBM over
registry, corpus, git, contagion and pretrained-sequence features, trained
with equal positive mass per campaign.

Extra training labels: the takeovers mined from deleted versions
(build/mine_unreported.py) are added as positives unless --no-mined.

Usage: python -m takeover_radar.train_final [--period week] [--no-mined]
"""

import argparse
import json

import numpy as np
import pandas as pd

from takeover_radar.experiments import add_mined_labels, load_mined
from takeover_radar.models import gbm
from takeover_radar.models.features import feature_cols
from takeover_radar.paths import INTERIM, MODEL_DIR, PROCESSED

GROUPS = ("registry", "corpus", "git", "contagion", "seq")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["month", "week"], default="month")
    ap.add_argument("--no-mined", action="store_true")
    args = ap.parse_args()
    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    weekly = args.period == "week"
    panel = pd.read_parquet(PROCESSED / ("panel_weekly.parquet" if weekly else "panel.parquet"))
    panel = panel[panel["in_universe"] == 1].reset_index(drop=True)
    panel = panel.merge(pd.read_parquet(PROCESSED / "seq_feats.parquet"), on=["name", "month"], how="left")
    if weekly:
        panel = panel.rename(columns={"month": "cal_month", "week": "month"})
    train_events, n_mined = events, 0
    if not args.no_mined:
        mined = load_mined(events, args.period)
        panel = add_mined_labels(panel, mined)
        train_events = pd.concat([events, mined[["name", "campaign"]]], ignore_index=True)
        n_mined = len(mined)
    feats = feature_cols(panel, GROUPS)
    model = gbm.fit(panel, feats, train_events, weight_mode="equal")
    model.save_model(str(MODEL_DIR / "takeover_radar_gbm.txt"))
    last = panel[panel["month"] == panel["month"].max()]
    np.save(MODEL_DIR / "reference_scores.npy", np.sort(gbm.predict(model, last, feats)))
    np.save(MODEL_DIR / "reference_popularity.npy", np.sort(gbm.popularity_score(last).to_numpy()))
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(MODEL_DIR / "feature_importance.csv", header=["gain"])
    (MODEL_DIR / "model_card.json").write_text(json.dumps({
        "score": "blend_pre_eq = mean(percentile(popularity), percentile(gbm))",
        "features": feats, "train_rows": len(panel), "train_events": int(panel["y"].sum()),
        "weight_mode": "equal", "period": args.period, "mined_labels": n_mined,
        "reference_month": int(last["cal_month" if weekly else "month"].iloc[0]),
        "reference_packages": len(last),
    }, indent=1))
    print(f"saved model ({len(feats)} features, {int(panel['y'].sum())} events); "
          f"reference distribution over {len(last)} packages")
    print(imp.head(20).round(1).to_string())


if __name__ == "__main__":
    main()
