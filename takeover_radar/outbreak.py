"""
Outbreak early warning: once the first victims of a campaign are publicly
reported, rank every other package by how likely it is to be (or become) a
victim of the same campaign.

Worms such as Shai-Hulud spread through stolen maintainer tokens (an infected
maintainer's every package gets republished) and through infected developer
machines (packages whose authors installed a victim). So proximity to the
reported victims in the maintainer graph, npm scope, GitHub organisation and
dependency graph carries most of the signal; the slow-moving takeover risk
from the monthly model is added as a prior.

Live publishes: in deployment the registry feed is public, so every version
published between the campaign start and tau is visible at tau. Worm victims
are new releases, often hours before anyone reports them, so publishing
activity since the campaign began (by anyone, and by maintainers tied to known
victims) is used too. Nothing published after tau is used.

Protocol: for every campaign with >= MIN_VICTIMS victims in the universe, at
checkpoints tau_k = time the k-th victim was reported (k in CHECKPOINTS), score
all at-risk packages not yet reported. Positives = victims of that campaign
not yet reported at tau_k. Leave-one-campaign-out training.

Usage: python -m takeover_radar.outbreak
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import timedelta
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from takeover_radar.build.panel import load, month_of, ts
from takeover_radar.collect.git_history import normalize_repo
from takeover_radar.paths import INTERIM, PROCESSED, ROOT, UNIVERSE

MIN_VICTIMS = 5
POST_WINDOW = timedelta(days=30)   # publishes kept per campaign for the live features
CHECKPOINTS = (1, 3, 5, 10, 25)
RESULTS = ROOT / "results"

_S = {}


def _init(cuts):
    _S["cuts"] = cuts


def post_publishes(doc, cut):
    """(time, publisher) of every version published in [cut, cut + POST_WINDOW).

    npm removed almost every malicious version after the fact, so they are gone
    from doc["versions"]; the time map still holds their publish times. Their
    publisher is lost (None). The "unpublished" list is never used: it records
    the cleanup after the report, which is not visible at tau."""
    pub = {v["v"]: v.get("pub") for v in doc["versions"]}
    out = []
    for ver, t in (doc.get("time") or {}).items():
        if ver in ("created", "modified"):
            continue
        t = ts(t)
        if t and cut <= t < cut + POST_WINDOW:
            out.append((t, pub.get(ver)))
    return sorted(out, key=lambda x: x[0])


def _state(name):
    """Latest legitimate state of a package before each campaign start."""
    doc = load(name)
    if doc is None or not doc["versions"]:
        return name, None
    vs = doc["versions"]
    out = {}
    for c, cut in _S["cuts"].items():
        prior = [v for v in vs if v["_t"] < cut]
        if len(prior) < 3:
            continue
        last = prior[-1]
        repo = normalize_repo(last.get("repo") or doc.get("repo"))
        out[c] = {"maint": tuple(last["maint"]), "pub": last.get("pub"),
                  "deps": tuple(last["deps"]),
                  "org": "/".join(repo.split("/")[:2]) if repo else None,
                  "repo": repo,
                  "post": post_publishes(doc, cut)}
    return name, out


def build_states(names, cuts, workers):
    path = INTERIM / "outbreak_states.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    states = defaultdict(dict)
    with Pool(workers, initializer=_init, initargs=(cuts,)) as pool:
        for k, (n, st) in enumerate(pool.imap_unordered(_state, names, chunksize=64), 1):
            if st:
                for c, s in st.items():
                    states[c][n] = s
            if k % 20000 == 0:
                print(f"  states {k}/{len(names)}", flush=True)
    states = dict(states)
    with open(path, "wb") as f:
        pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)
    return states


def checkpoint_rows(camp, k, victims, st, prior, cut):
    """Feature rows for every candidate at checkpoint k of campaign camp."""
    victims = victims.sort_values("t_reported")
    tau = victims["t_reported"].iloc[k - 1]
    known = set(victims.loc[victims["t_reported"] <= tau, "name"])
    target = set(victims["name"]) - known
    by_maint, by_org, by_scope = defaultdict(int), defaultdict(int), defaultdict(int)
    known_set = set()
    for n in known:
        s = st.get(n)
        if s is None:
            continue
        known_set.add(n)
        for m in s["maint"]:
            by_maint[m] += 1
        if s["org"]:
            by_org[s["org"]] += 1
        if n.startswith("@"):
            by_scope[n.split("/")[0]] += 1
    known_maint = set(by_maint)
    victim_deps = set()
    for n in known_set:
        victim_deps.update(st[n]["deps"])
    # second hop: maintainers who co-maintain anything with a known victim's maintainer
    hop2 = set()
    for n, s in st.items():
        if known_maint.intersection(s["maint"]):
            hop2.update(s["maint"])
    hop2 -= known_maint
    # Worms republish in tight bursts: every publish visible at tau, across all candidates,
    # and the malicious publish times of the victims reported so far.
    all_pubs = np.sort(np.array([t.timestamp() for s in st.values()
                                 for t, _ in s.get("post", ()) if t <= tau]))
    known_t = np.sort(victims.loc[victims["name"].isin(known), "t_event"]
                      .map(lambda t: t.timestamp()).to_numpy())
    rows = []
    for n, s in st.items():
        if n in known:
            continue
        maint = s["maint"]
        post = [(t, pb) for t, pb in s.get("post", ()) if t <= tau]
        if post:
            lt = post[-1][0].timestamp()
            burst30 = np.searchsorted(all_pubs, lt + 1800, "right") - np.searchsorted(all_pubs, lt - 1800) - 1
            burst6h = np.searchsorted(all_pubs, lt + 21600, "right") - np.searchsorted(all_pubs, lt - 21600) - 1
            i = np.searchsorted(known_t, lt)
            near = min([abs(lt - known_t[j]) for j in (i - 1, i) if 0 <= j < len(known_t)], default=np.nan)
        else:
            burst30 = burst6h = 0
            near = np.nan
        f = {
            "name": n, "campaign": camp, "k": k, "y": int(n in target),
            "o_comaint_victims": sum(by_maint.get(m, 0) for m in maint),
            "o_shared_maint": sum(m in known_maint for m in maint),
            "o_shared_maint_frac": (sum(m in known_maint for m in maint) / len(maint)) if maint else 0.0,
            "o_pub_is_victim_maint": float(s["pub"] in known_maint),
            "o_hop2_maint": sum(m in hop2 for m in maint),
            "o_org_victims": by_org.get(s["org"], 0) if s["org"] else 0,
            "o_scope_victims": by_scope.get(n.split("/")[0], 0) if n.startswith("@") else 0,
            "o_deps_on_victim": sum(d in known_set for d in s["deps"]),
            "o_victim_depends_on": float(n in victim_deps),
            # Publisher-based counts are left out on purpose: removed (malicious) versions
            # lost their publisher, so such counts would differ between victims and others
            # for a reason a live feed would not reproduce.
            "o_pubs_since_start": len(post),
            "o_hours_since_pub": ((tau - post[-1][0]).total_seconds() / 3600) if post else 1e4,
            "o_hours_start_to_pub": ((post[0][0] - cut).total_seconds() / 3600) if post else 1e4,
            "o_burst_30m": burst30,
            "o_burst_6h": burst6h,
            "o_hours_to_victim_pub": near / 3600 if near == near else 1e4,
            "o_n_maint": len(maint),
            "o_known": len(known),
            "o_prior": prior.get(n, np.nan),
        }
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    events["t_event"] = pd.to_datetime(events["t_event"], format="ISO8601")
    events["t_reported"] = pd.to_datetime(events["t_reported"], utc=True, format="ISO8601").dt.tz_localize(None)
    # A report can never precede the malicious publish; clamp data-entry noise.
    events["t_reported"] = events[["t_reported", "t_event"]].max(axis=1)
    with open(UNIVERSE, encoding="utf-8") as f:
        universe = list(dict.fromkeys(json.loads(l)["name"] for l in f))
    uni = set(universe)
    ev_u = events[events["name"].isin(uni)]
    camps = ev_u["campaign"].value_counts()
    camps = camps[camps >= MIN_VICTIMS].index.tolist()
    cuts = {c: events.loc[events["campaign"] == c, "t_event"].min().to_pydatetime() - timedelta(seconds=1)
            for c in camps}
    print(f"{len(camps)} campaigns with >= {MIN_VICTIMS} universe victims", flush=True)
    states = build_states(universe, cuts, args.workers)

    # Prior: monthly-model risk percentile in the campaign's month (out-of-fold / held-out preds).
    prior_by_month = {}
    for p in sorted(RESULTS.glob("preds_*gbm_full*.parquet")):
        if "smoke" in p.name:
            continue
        d = pd.read_parquet(p)
        if "gbm_full" not in d:
            continue
        d["pr"] = d.groupby("month")["gbm_full"].rank(pct=True)
        for m, g in d.groupby("month"):
            prior_by_month[m] = dict(zip(g["name"], g["pr"]))

    frames = []
    for c in camps:
        vic = ev_u[ev_u["campaign"] == c]
        st = states.get(c, {})
        prior = prior_by_month.get(month_of(cuts[c]), {})
        for k in CHECKPOINTS:
            if k >= len(vic):
                break
            frames.append(checkpoint_rows(c, k, vic, st, prior, cuts[c]))
        print(f"  campaign {c}: {len(vic)} victims, {len(st)} candidates", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(PROCESSED / "outbreak.parquet", index=False)
    feats = [c for c in data.columns if c.startswith("o_")]

    # Leave-one-campaign-out
    data["score"] = np.nan
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=20)
    for c in camps:
        te = data["campaign"] == c
        tr = data[~te]
        # balance campaigns: each contributes equal positive mass
        w = np.ones(len(tr))
        pos = tr["y"].to_numpy() == 1
        cnt = tr.loc[pos, "campaign"].map(tr.loc[pos, "campaign"].value_counts()).to_numpy()
        w[pos] = 1.0 / cnt
        w[pos] *= pos.sum() / w[pos].sum()
        m = lgb.train(params, lgb.Dataset(tr[feats], tr["y"], weight=w), 300)
        data.loc[te, "score"] = m.predict(data.loc[te, feats])
    data["heur_comaint"] = data["o_comaint_victims"] + 0.5 * data["o_org_victims"] + \
        0.5 * data["o_scope_victims"] + 0.01 * data["o_prior"].fillna(0)
    data["prior_only"] = data["o_prior"].fillna(0)

    rows = []
    for (c, k), g in data.groupby(["campaign", "k"]):
        npos = int(g["y"].sum())
        if npos == 0:
            continue
        for col in ("score", "heur_comaint", "prior_only"):
            order = g.sort_values(col, ascending=False)["y"].to_numpy()
            rec = {f"recall@{n}": order[:n].sum() / npos for n in (10, 100, 1000)}
            from sklearn.metrics import roc_auc_score
            rows.append({"campaign": c, "k": k, "model": col, "remaining": npos,
                         "candidates": len(g), "auc": roc_auc_score(g["y"], g[col]), **rec})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "outbreak_by_campaign.csv", index=False)
    summ = res.groupby(["model", "k"])[["auc", "recall@10", "recall@100", "recall@1000"]].mean()
    print(summ.round(3).to_string())
    summ.reset_index().to_csv(RESULTS / "outbreak_summary.csv", index=False)


if __name__ == "__main__":
    main()
