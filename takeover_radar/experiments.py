"""
Run every model under both protocols and write results/.

  dev   campaign-grouped 5-fold CV on months before 2026-01 (model selection)
  test  train on months before 2026-01, score 2026-01..2026-10 (held out; run once)

Usage:
  python -m takeover_radar.experiments --split dev --models popularity,weak_links,gbm_full
  python -m takeover_radar.experiments --split test --models all
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from takeover_radar.eval.protocol import campaign_folds, event_percentiles, per_campaign, summarize
from takeover_radar.models import gbm, seq
from takeover_radar.models.features import DEV_END, TEST_START, feature_cols
from takeover_radar.paths import INTERIM, PROCESSED, ROOT

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURE_SETS = {
    "static": ("registry", "corpus"),
    "git": ("registry", "corpus", "git"),
    "full": ("registry", "corpus", "git", "contagion"),
    "no_contagion": ("registry", "corpus", "git"),
    "pre": ("registry", "corpus", "git", "contagion", "seq"),
}

ALL = ["popularity", "weak_links", "logistic", "gbm_static", "gbm_git", "gbm_full",
       "seq_full", "ensemble"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_model(name, train, test, events, val=None):
    if name == "popularity":
        return gbm.popularity_score(test)
    if name == "weak_links":
        return gbm.weak_links_score(test)
    if name == "logistic":
        return gbm.logistic(train, test, feature_cols(train, FEATURE_SETS["full"]), events)
    if name.startswith("gbm_"):
        # gbm_<set>[_eq]: _eq gives every campaign equal positive mass instead of sqrt
        fs, eq = (name[4:-3], True) if name.endswith("_eq") else (name[4:], False)
        feats = feature_cols(train, FEATURE_SETS[fs])
        model = gbm.fit(train, feats, events, weight_mode="equal" if eq else "sqrt")
        return gbm.predict(model, test, feats)
    if name.startswith("seq_"):
        feats = feature_cols(train, FEATURE_SETS[name[4:]])
        model, stats = seq.fit(train, feats, events, val_df=val, log=log)
        return seq.score(model, stats, test, feats)
    raise ValueError(name)


def rank_avg(*scores_by_month):
    """Average of within-month percentile ranks (ensembling models on different scales)."""
    return np.mean(scores_by_month, axis=0)


def within_month_rank(df, col):
    return df.groupby("month")[col].rank(pct=True).to_numpy()


def inner_val_split(train, events, seed=0):
    f = campaign_folds(train["name"], events, k=8, seed=seed + 100,
                       positives=train.loc[train["y"] == 1, "name"])
    is_val = train["name"].map(f) == 0
    return train[~is_val], train[is_val]


def load_mined(events, period):
    """Mined (unreported) takeovers as extra events with their own campaign ids."""
    from takeover_radar.build.labels import CAMPAIGN_GAP_DAYS
    from takeover_radar.build.panel import month_of, week_of
    m = pd.read_json(INTERIM / "mined_events.jsonl", lines=True)
    m = m[~m["name"].isin(set(events["name"]))]
    t = pd.to_datetime(m["t_event"], format="ISO8601")
    m["period"] = t.map(week_of if period == "week" else month_of)
    m = m.assign(t=t).sort_values("t")
    gap = m["t"].diff() > pd.Timedelta(days=CAMPAIGN_GAP_DAYS)
    m["campaign"] = 100_000 + gap.cumsum().to_numpy()
    return m.drop(columns="t").reset_index(drop=True)


def add_mined_labels(train, mined):
    """y = 1 in the mined event period; the package leaves the risk set after it."""
    ev = train["name"].map(mined.set_index("name")["period"])
    keep = ev.isna() | (train["month"] <= ev)
    train = train[keep].copy()
    ev = ev[keep]
    train.loc[train["month"] == ev, "y"] = 1
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--models", default="popularity,weak_links,gbm_full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--period", choices=["month", "week"], default="month",
                    help="week: weekly panel; risk sets, offsets and metrics are per week")
    ap.add_argument("--mined", choices=["off", "exclude", "train"], default="off",
                    help="exclude: drop mined-takeover packages from evaluation; "
                         "train: also add them as positives to training")
    args = ap.parse_args()
    args.panel = args.panel or str(PROCESSED / ("panel_weekly.parquet" if args.period == "week"
                                                else "panel.parquet"))
    models = ALL if args.models == "all" else args.models.split(",")

    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    panel = pd.read_parquet(args.panel)
    panel = panel[panel["in_universe"] == 1].reset_index(drop=True)
    if any("pre" in m for m in models):
        sf = pd.read_parquet(PROCESSED / "seq_feats.parquet")
        panel = panel.merge(sf, on=["name", "month"], how="left")
    if args.period == "week":
        # every model groups by "month"; on the weekly panel that column becomes the week
        panel = panel.rename(columns={"month": "cal_month", "week": "month"})
    else:
        panel["cal_month"] = panel["month"]
    log(f"panel {len(panel)} rows, {panel['name'].nunique()} packages, {int(panel['y'].sum())} events")
    mined, train_events = None, events
    if args.mined != "off":
        mined = load_mined(events, args.period)
        train_events = pd.concat([events, mined[["name", "campaign"]]], ignore_index=True)
        log(f"mined takeovers: {len(mined)} ({args.mined})")

    def prep(train, test):
        if mined is None:
            return train, test
        test = test[~test["name"].isin(set(mined["name"]))]
        if args.mined == "train":
            train = add_mined_labels(train, mined)
        return train, test

    preds = {}
    if args.split == "dev":
        data = panel[panel["cal_month"] < DEV_END].reset_index(drop=True)
        fold = data["name"].map(campaign_folds(data["name"], events, k=args.folds,
                                                positives=data.loc[data["y"] == 1, "name"]))
        for m in models:
            if m == "ensemble" or m.startswith("blend_"):
                continue
            s = np.full(len(data), np.nan)
            for k in range(args.folds):
                te = (fold == k).to_numpy()
                train, test = prep(data[~te], data[te])
                tr, va = inner_val_split(train, events) if m.startswith("seq_") else (train, None)
                log(f"{m} fold {k}: train {len(tr)} rows / {int(tr['y'].sum())} events, "
                    f"test {int(test['y'].sum())} events")
                idx = np.where(te)[0][data[te]["name"].isin(test["name"]).to_numpy()]
                s[idx] = run_model(m, tr, test, train_events, val=va)
            preds[m] = s
    else:
        train = panel[panel["cal_month"] < DEV_END].reset_index(drop=True)
        data = panel[panel["cal_month"] >= TEST_START].reset_index(drop=True)
        train, data = prep(train, data)
        data = data.reset_index(drop=True)
        for m in models:
            if m == "ensemble" or m.startswith("blend_"):
                continue
            tr, va = inner_val_split(train, events) if m.startswith("seq_") else (train, None)
            log(f"{m}: train {len(tr)} rows / {int(tr['y'].sum())} events, test {int(data['y'].sum())} events")
            preds[m] = run_model(m, tr, data, train_events, val=va)

    if "ensemble" in models and {"gbm_full", "seq_full"} <= preds.keys():
        tmp = data[["month"]].copy()
        tmp["a"], tmp["b"] = preds["gbm_full"], preds["seq_full"]
        preds["ensemble"] = rank_avg(within_month_rank(tmp, "a"), within_month_rank(tmp, "b"))

    # blend_<set>[_eq]: popularity prior and gbm_<set>[_eq], equal within-month rank weight
    for m in models:
        if m.startswith("blend_") and f"gbm_{m[6:]}" in preds:
            tmp = data[["month"]].assign(a=gbm.popularity_score(data).to_numpy(), b=preds[f"gbm_{m[6:]}"])
            preds[m] = rank_avg(within_month_rank(tmp, "a"), within_month_rank(tmp, "b"))

    if mined is not None and args.split == "dev":
        keep = ~data["name"].isin(set(mined["name"])).to_numpy()
        data = data[keep].reset_index(drop=True)
        preds = {m: s[keep] for m, s in preds.items()}

    out, tables = {}, {}
    for m, s in preds.items():
        df = data[["name", "month", "y"]].assign(score=s)
        ev = event_percentiles(df)
        out[m] = summarize(ev, events)
        tables[m] = per_campaign(ev, events)
        log(m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[m].items()}))
    tag = f"{args.split}{('_' + args.tag) if args.tag else ''}"
    prev = RESULTS / f"metrics_{tag}.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    merged.update(out)
    prev.write_text(json.dumps(merged, indent=1, default=float))
    for m, t in tables.items():
        t.to_csv(RESULTS / f"campaigns_{tag}_{m}.csv")
    pd.DataFrame({m: s for m, s in preds.items()}).assign(
        name=data["name"], month=data["month"], y=data["y"]).to_parquet(
        RESULTS / f"preds_{tag}_{'_'.join(preds)}.parquet")


if __name__ == "__main__":
    main()
