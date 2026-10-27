"""
Mine unreported takeovers from the packument time map.

npm removes malicious versions after the fact, but the `time` map keeps their
publish timestamps. So every package is scanned for *deleted versions*: in
`time`, absent from `versions`. Authors also unpublish broken releases, so
each deleted version gets takeover-like evidence and is scored against the
deletions that OSV already labels as takeovers (events.jsonl):

  * legitimate history first (same rule as labels.py: >= 3 prior versions,
    >= 90 days old, a live prior version)
  * cross-package burst: other packages deleting versions within hours,
    especially packages sharing a maintainer (stolen token / worm)
  * dormancy before the deleted publish, quick live fix release after it
  * the deleted version was the newest at publish time (attackers bump the
    latest line so installs pick it up)

Nothing here uses the "unpublished" list order or removal time, which are
not in the data anyway; the score is for labelling only, never a model feature.

Outputs
  data/interim/deleted_versions.parquet   one row per deleted version + score
  data/interim/mined_events.jsonl         high-confidence candidates not in OSV,
                                          in events.jsonl format

Usage: python -m takeover_radar.build.mine_unreported --workers 10
"""

import argparse
import json
from collections import defaultdict
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from takeover_radar.build.labels import CAMPAIGN_GAP_DAYS, MIN_PRIOR_DAYS, MIN_PRIOR_VERSIONS, PROTESTWARE
from takeover_radar.build.panel import load, ts
from takeover_radar.paths import INTERIM, PACKUMENTS, ROOT, UNIVERSE

RESULTS = ROOT / "results"
HOUR = 3600.0


def _semver(v):
    core = str(v).split("+")[0]
    pre = "-" in core
    parts = core.split("-")[0].split(".")
    try:
        return tuple(int(p) for p in parts[:3]) + (0,) * (3 - len(parts[:3])), pre
    except ValueError:
        return None, pre


def scan(path):
    import gzip
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return []
    if doc.get("missing") or not doc.get("time"):
        return []
    name = doc["name"]
    live = {v["v"]: v for v in doc.get("versions", [])}
    allt = sorted((ts(t), v) for v, t in doc["time"].items()
                  if v not in ("created", "modified", "unpublished") and ts(t))
    deleted = [(t, v) for t, v in allt if v not in live]
    if not deleted:
        return []
    from bisect import bisect_left, bisect_right
    live_sorted = sorted(((ts(v["t"]), v) for v in live.values() if ts(v.get("t"))),
                         key=lambda x: x[0])
    live_t = [x[0] for x in live_sorted]
    all_t = [x[0] for x in allt]
    # running max semver over the full time-ordered history (prefix max)
    top_before, cur = [], None
    for _, v in allt:
        top_before.append(cur)
        s = _semver(v)[0]
        if s and (cur is None or s > cur):
            cur = s
    out = []
    for t, ver in deleted:
        i = bisect_left(all_t, t)
        n_prior = i
        il = bisect_left(live_t, t)
        ja = bisect_right(live_t, t)
        sv, pre = _semver(ver)
        top = top_before[i] if i < len(top_before) else cur
        last_live = live_sorted[il - 1][1] if il else None
        nxt = live_sorted[ja] if ja < len(live_sorted) else None
        bump = None
        if sv and top:
            bump = ("major" if sv[0] > top[0] else "minor" if sv[:2] > top[:2]
                    else "patch" if sv > top else "old_line")
        out.append({
            "name": name, "version": ver, "t": t,
            "n_prior": n_prior, "n_prior_live": il,
            "age_days": (t - allt[0][0]).total_seconds() / 86400 if n_prior else 0.0,
            "dormant_days": (t - allt[i - 1][0]).total_seconds() / 86400 if n_prior else np.nan,
            "next_live_hours": (nxt[0] - t).total_seconds() / HOUR if nxt else np.nan,
            "next_live_pub_same": float(bool(nxt and last_live and nxt[1].get("pub") == last_live.get("pub"))),
            "bump": bump, "prerelease": pre,
            "n_deleted_pkg": len(deleted), "n_live_pkg": len(live),
            "fully_unpublished": bool(doc.get("fully_unpublished")),
            "maint": tuple(last_live["maint"]) if last_live else (),
            "pub_last": last_live.get("pub") if last_live else None,
            "hooks_last": bool(last_live and last_live.get("hooks")),
        })
    return out


def _secs(t):
    """Epoch seconds, whatever the datetime64 unit (parquet round-trips as [us])."""
    return (t - pd.Timestamp(0)).dt.total_seconds().to_numpy()


def burst_features(df):
    """Cross-package clustering of deletions in time and in the maintainer graph."""
    df = df.sort_values("t").reset_index(drop=True)
    te = _secs(df["t"])
    # one entry per package-time, so a package deleting 10 versions is not a "burst"
    first_per_pkg_day = df.groupby(["name", df["t"].dt.floor("D")]).head(1)
    pt = np.sort(_secs(first_per_pkg_day["t"]))
    for w, lab in ((6 * HOUR, "6h"), (48 * HOUR, "48h")):
        df[f"burst_{lab}"] = (np.searchsorted(pt, te + w, "right") - np.searchsorted(pt, te - w, "left") - 1)
    # same-maintainer burst: other packages of any shared maintainer with a deletion within 7 days
    by_maint = defaultdict(list)
    for i, (n, ms, t) in enumerate(zip(df["name"], df["maint"], te)):
        for m in ms:
            by_maint[m].append((t, n))
    keys = {}
    for m in by_maint:
        by_maint[m].sort()
        keys[m] = np.array([x[0] for x in by_maint[m]])
    co = np.zeros(len(df))
    for i, (n, ms, t) in enumerate(zip(df["name"], df["maint"], te)):
        hits = set()
        for m in ms:
            lst = by_maint[m]
            if len(lst) > 5000:       # npm bot / org accounts: uninformative and slow
                continue
            lo = np.searchsorted(keys[m], t - 7 * 86400)
            for tt, nn in lst[lo:]:
                if tt > t + 7 * 86400:
                    break
                if nn != n:
                    hits.add(nn)
        co[i] = len(hits)
    df["comaint_deleting_7d"] = co
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="score threshold for mined labels (default: the value giving the "
                         "same precision proxy as OSV events, chosen below)")
    args = ap.parse_args()

    out_pq = INTERIM / "deleted_versions_raw.parquet"
    if out_pq.exists():
        df = pd.read_parquet(out_pq)
    else:
        files = sorted(PACKUMENTS.glob("*.json.gz"))
        rows = []
        with Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(scan, files, chunksize=128), 1):
                rows.extend(r)
                if k % 10000 == 0 or k == len(files):
                    print(f"  scanned {k}/{len(files)} deleted={len(rows)}", flush=True)
        df = pd.DataFrame(rows)
        df["maint"] = df["maint"].map(list)
        df.to_parquet(out_pq, index=False)
    df["maint"] = df["maint"].map(tuple)
    print(f"{len(df)} deleted versions in {df['name'].nunique()} packages", flush=True)

    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    osv_names = set()
    with open(INTERIM / "osv_reports.jsonl", encoding="utf-8") as f:
        for line in f:
            osv_names.add(json.loads(line)["name"])
    with open(UNIVERSE, encoding="utf-8") as f:
        uni = {json.loads(l)["name"] for l in f}
    mal = {(r.name, v) for r in events.itertuples() for v in r.malicious_versions}
    df["label_osv_event"] = [int((n, v) in mal) for n, v in zip(df["name"], df["version"])]
    df["osv_any"] = df["name"].isin(osv_names).astype(int)
    df["in_universe"] = df["name"].isin(uni).astype(int)

    df = burst_features(df)
    df["eligible"] = ((df["n_prior"] >= MIN_PRIOR_VERSIONS) & (df["age_days"] >= MIN_PRIOR_DAYS)
                      & (df["n_prior_live"] >= 1) & ~df["fully_unpublished"]
                      & ~df["name"].isin(PROTESTWARE)
                      # npm security placeholders (0.0.1-security) are registry actions, not publishes
                      & ~df["version"].str.endswith("-security")).astype(int)

    # Score: a small GBM separating OSV-labelled malicious deletions from all other
    # eligible deletions, out-of-fold by package (so a package never scores itself).
    import lightgbm as lgb
    feats = ["dormant_days", "next_live_hours", "next_live_pub_same", "prerelease", "n_deleted_pkg",
             "n_live_pkg", "n_prior", "age_days", "burst_6h", "burst_48h", "comaint_deleting_7d",
             "hooks_last", "bump_major", "bump_minor", "bump_patch", "bump_old"]
    for b in ("major", "minor", "patch"):
        df[f"bump_{b}"] = (df["bump"] == b).astype(float)
    df["bump_old"] = (df["bump"] == "old_line").astype(float)
    df["prerelease"] = df["prerelease"].astype(float)
    df["hooks_last"] = df["hooks_last"].astype(float)
    el = df[df["eligible"] == 1].copy()
    # OSV packages that are *not* takeover events (e.g. malicious from day one but with a
    # long history of spam) are left out of the negatives: their label is uncertain.
    el = el[~((el["osv_any"] == 1) & (el["label_osv_event"] == 0))]
    camp = events.set_index("name")["campaign"]
    rng = np.random.default_rng(0)
    # fold by campaign for positives, by package for the rest (no worm-wave leakage)
    pk = el["name"].unique()
    pfold = dict(zip(pk, rng.integers(0, 5, len(pk))))
    cf = {c: int(rng.integers(0, 5)) for c in camp.unique()}
    el["fold"] = [cf[camp[n]] if n in camp.index else pfold[n] for n in el["name"]]
    el["score"] = np.nan
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                  feature_fraction=0.8, lambda_l2=10.0, verbose=-1, num_threads=12)
    for k in range(5):
        tr, te = el["fold"] != k, el["fold"] == k
        m = lgb.train(params, lgb.Dataset(el.loc[tr, feats], el.loc[tr, "label_osv_event"]), 300)
        el.loc[te, "score"] = m.predict(el.loc[te, feats])
    final = lgb.train(params, lgb.Dataset(el[feats], el["label_osv_event"]), 300)
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = el["label_osv_event"]
    print(f"eligible deletions {len(el)}, OSV takeover deletions {int(y.sum())}")
    print(f"OOF AUC {roc_auc_score(y, el['score']):.3f}  AP {average_precision_score(y, el['score']):.3f} "
          f"(base rate {y.mean():.4f})")
    print(imp.round(0).to_string())

    # Per package: the first deletion with the highest score is the candidate takeover.
    el = el.sort_values("score", ascending=False)
    pkg = el.groupby("name").head(1).copy()
    pkg["is_osv"] = pkg["name"].isin(set(events["name"])).astype(int)
    # Threshold: the lowest score at which the *local* OSV share (centred band of 250
    # ranked packages) is still >= 0.5. Unreported takeovers count as negatives in that
    # share, so the true precision at the threshold is higher than 0.5.
    s = pkg.sort_values("score", ascending=False)
    band = s["is_osv"].rolling(250, center=True, min_periods=50).mean().to_numpy()
    ok = np.where(band >= 0.5)[0]
    thr = args.threshold or float(s["score"].to_numpy()[ok.max()])
    for q in (100, 250, 500, 1000, 2500, 5000):
        print(f"  top {q:5d} packages: OSV share {s['is_osv'].head(q).mean():.3f}")
    cand = pkg[(pkg["score"] >= thr) & (pkg["is_osv"] == 0)].copy()
    print(f"threshold {thr:.3f}: {int((pkg['score'] >= thr).sum())} packages above, "
          f"{len(cand)} not in OSV ({int(cand['in_universe'].sum())} in universe)")

    el.drop(columns=["maint"]).to_parquet(INTERIM / "deleted_versions.parquet", index=False)
    show = ["name", "version", "t", "score", "dormant_days", "next_live_hours", "burst_6h",
            "comaint_deleting_7d", "bump", "in_universe"]
    cand[show].to_csv(RESULTS / "mined_candidates.csv", index=False)
    print(cand[show].head(40).to_string())

    # events.jsonl-format records for the candidates (campaign ids assigned downstream)
    with open(INTERIM / "mined_events.jsonl", "w", encoding="utf-8") as f:
        for r in cand.itertuples():
            vs = el[(el["name"] == r.name) & (el["t"] >= r.t) & (el["t"] <= r.t + timedelta(days=2))]
            f.write(json.dumps({
                "name": r.name, "t_event": r.t.isoformat(),
                "malicious_versions": sorted(vs["version"].unique()),
                "n_prior_versions": int(r.n_prior), "first_publish": None,
                "report_ids": ["MINED"], "t_reported": None,
                "summary": f"mined: deleted version, score {r.score:.3f}",
                "mined_score": float(r.score),
            }) + "\n")


if __name__ == "__main__":
    main()
