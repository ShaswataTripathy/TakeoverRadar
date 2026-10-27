"""
Stress test: does the model react to a slow social takeover (the xz pattern)?

Real npm data has only a few slow takeovers (event-stream), so the pattern is
injected into the *raw* histories of real, never-compromised packages and
features are recomputed with the production feature code:

  month 0-5  a brand-new contributor ramps up commits (2 -> 20 per month);
             from month 2 they touch CI/build files, in month 4 binary test
             fixtures (where xz hid its payload)
  month 1+   the historically dominant author stops committing
  month 3    the old maintainer publishes a release adding the newcomer as maintainer
  month 5    the newcomer publishes a release themselves: new tooling, no provenance
  month 6    (payload would ship here; never observed by the model)

Variants isolate the channels: `git` (contributor dynamics only), `registry`
(maintainer + publisher changes only) and `full`. Each synthetic history is
compared with the same package left untouched, as a percentile within the real
at-risk population of that month.

Usage: python -m takeover_radar.eval.synthetic_xz --n 400
"""

import argparse
import copy
import gzip
import json
import random
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from takeover_radar.build import panel as P
from takeover_radar.collect.git_history import BUILD, CI, FIXTURE, SRC, normalize_repo, out_path
from takeover_radar.models import gbm
from takeover_radar.models.features import DEV_END, feature_cols
from takeover_radar.models.pretrain import load_encoder, seq_features
from takeover_radar.paths import INTERIM, MODEL_DIR, PROCESSED, ROOT

T0_MONTH = DEV_END                     # 2026-01, inside the held-out test period
HORIZON = 8
RAMP = [2, 4, 8, 12, 15, 20]


def load_model(panel, events):
    """The deployed GBM component (gbm_pre_eq), trained on dev months only."""
    path = MODEL_DIR / "gbm_pre_eq_dev.txt"
    feats = feature_cols(panel, ("registry", "corpus", "git", "contagion", "seq"))
    if path.exists():
        return lgb.Booster(model_file=str(path)), feats
    model = gbm.fit(panel[panel["month"] < DEV_END], feats, events, weight_mode="equal")
    model.save_model(str(path))
    return model, feats


def inject(doc, commits, scenario, rng):
    """Return modified (doc, commits) for one scenario."""
    T0 = P.MONTH_STARTS[T0_MONTH]
    doc = copy.deepcopy(doc)
    commits = [list(c) for c in commits]
    newcomer_git = "%012x" % rng.getrandbits(48)
    newcomer_npm = f"synthetic-{newcomer_git[:8]}"
    if scenario in ("git", "full"):
        hist = [c for c in commits if c[0] < T0.timestamp()]
        top = pd.Series([c[2] for c in hist]).value_counts().index[0]
        fade = (T0 + timedelta(days=30)).timestamp()
        commits = [c for c in commits if not (c[2] == top and c[0] >= fade)]
        for mo, n in enumerate(RAMP):
            start = P.MONTH_STARTS[T0_MONTH + mo].timestamp()
            for i in range(n):
                t = int(start + rng.random() * 28 * 86400)
                mask = SRC
                if mo >= 2 and i % 2 == 0:
                    mask |= CI | BUILD
                if mo == 4 and i < 2:
                    mask |= FIXTURE
                # newcomer commits get applied by themselves (they have push access)
                commits.append([t, t, newcomer_git, newcomer_git, mask, 3, 0])
    if scenario in ("registry", "full"):
        last = doc["versions"][-1]
        t1 = P.MONTH_STARTS[T0_MONTH + 3] + timedelta(days=5)
        v1 = dict(copy.deepcopy(last), v="99.0.0-syn.1", t=t1.isoformat(), _t=t1,
                  maint=sorted(set(last["maint"]) | {newcomer_npm}))
        t2 = P.MONTH_STARTS[T0_MONTH + 5] + timedelta(days=3)
        npm_major = P._major(last.get("npmv"))
        v2 = dict(copy.deepcopy(v1), v="99.0.0-syn.2", t=t2.isoformat(), _t=t2, pub=newcomer_npm,
                  pub_dom="proton.me", tp=None, att=False,
                  npmv=f"{(npm_major + 2) if npm_major > 0 else 6}.0.0")
        doc["versions"] = [v for v in doc["versions"] if v["_t"] < t1] + [v1, v2]
        P._C["first_seen"][newcomer_npm] = t1
    commits.sort()
    return doc, commits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel = panel[panel["in_universe"] == 1].reset_index(drop=True)
    panel = panel.merge(pd.read_parquet(PROCESSED / "seq_feats.parquet"), on=["name", "month"], how="left")
    model, feats = load_model(panel, events)
    encoder, meta = load_encoder()
    P._init(str(INTERIM / "corpus.pkl"), str(INTERIM / "events.jsonl"))

    months = set(range(T0_MONTH, T0_MONTH + HORIZON))
    ref = panel[panel["month"].isin(months)].copy()
    ref["s"] = model.predict(ref[feats].to_numpy(np.float32), raw_score=True)
    ref_sorted = {m: np.sort(g["s"].to_numpy()) for m, g in ref.groupby("month")}

    # Eligible: never compromised, active repo, long history before T0.
    T0 = P.MONTH_STARTS[T0_MONTH]
    at_t0 = panel[(panel["month"] == T0_MONTH) & (panel["g_commits_365"] >= 5)
                  & (panel["g_repo_age_days"] >= 730) & (panel["r_days_since_pub"] <= 730)]
    pool = sorted(set(at_t0["name"]) - set(events["name"]))
    rng.shuffle(pool)

    out = []
    done = 0
    for name in pool:
        if done >= args.n:
            break
        doc = P.load(name)
        if doc is None or not doc["versions"]:
            continue
        repo = normalize_repo(doc["versions"][-1].get("repo") or doc.get("repo"))
        if not repo or not out_path(repo).exists():
            continue
        with gzip.open(out_path(repo), "rt", encoding="utf-8") as f:
            commits = json.load(f).get("commits") or []
        if not commits:
            continue
        for scenario in ("control", "git", "registry", "full"):
            if scenario == "control":
                d, c = doc, commits
            else:
                d, c = inject(doc, commits, scenario, rng)
            # full history: the sequence encoder reads every month up to the scored one
            rows = P.rows_from(name, d, P.git_arrays_from_commits(c))
            if not rows:
                break
            df = pd.DataFrame(rows)
            df = df.merge(seq_features(df, encoder, meta), on=["name", "month"], how="left")
            df = df[df["month"].isin(months)].reindex(columns=["name", "month"] + feats)
            if df.empty:
                break
            s = model.predict(df[feats].to_numpy(np.float32), raw_score=True)
            for m, sc in zip(df["month"], s):
                pct = np.searchsorted(ref_sorted[m], sc) / len(ref_sorted[m])
                out.append({"name": name, "scenario": scenario, "rel_month": m - T0_MONTH, "pct": pct})
        else:
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{args.n}", flush=True)

    res = pd.DataFrame(out)
    res.to_csv(ROOT / "results" / "synthetic_xz_raw.csv", index=False)
    summ = res.groupby(["scenario", "rel_month"])["pct"].agg(
        median="median", top5=lambda p: (p >= 0.95).mean(), top1=lambda p: (p >= 0.99).mean())
    print(summ.round(3).unstack(0).to_string())
    summ.reset_index().to_csv(ROOT / "results" / "synthetic_xz_summary.csv", index=False)


if __name__ == "__main__":
    main()
