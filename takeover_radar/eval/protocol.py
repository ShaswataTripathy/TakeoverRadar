"""
Evaluation protocol.

Takeovers come in campaigns (a worm wave can hit hundreds of packages in a
day), so two splits are used:

  * campaign-grouped K-fold: every package taken over in the same campaign lands
    in the same fold; never-compromised packages are spread at random. Nothing
    from a test campaign is seen in training.
  * temporal: train on months before a cutoff, test on months after it. This is
    the deployment setting: the model has only seen past campaigns.

Metrics are stratified by calendar month. The overall rate of takeovers swings
by orders of magnitude between months (worm waves), which says nothing about
which packages are at risk, so each event is scored only against packages at
risk in the same month:

  percentile  share of at-risk packages that month scoring below the victim
              (its mean over events is the month-stratified AUC / C-index)
  recall@k    share of victims ranked in the top k% of that month's risk set

Confidence intervals come from a cluster bootstrap over campaigns.
"""

import numpy as np
import pandas as pd

TOPK = (0.001, 0.01, 0.05)


def campaign_folds(names, events, k=5, seed=0, positives=None):
    """Map package name -> fold, keeping each campaign inside one fold.

    `positives`: packages whose takeover row is in the data. Folds are balanced on
    these only; a package taken over after the data window has no positive row
    here, and counting it left whole folds without a single test event."""
    rng = np.random.default_rng(seed)
    camp = events.set_index("name")["campaign"].to_dict()
    names = pd.Index(pd.unique(names))
    pos = [n for n in names if n in camp]
    counted = pos if positives is None else [n for n in pd.unique(positives) if n in camp]
    sizes = pd.Series([camp[n] for n in counted]).value_counts()
    sizes = sizes.reindex(list(dict.fromkeys(camp[n] for n in pos)), fill_value=0)
    load, cfold = np.zeros(k), {}
    order = list(sizes.index)
    rng.shuffle(order)
    for c in sorted(order, key=lambda c: -sizes[c]):    # largest first, then balance
        f = int(np.argmin(load))
        cfold[c] = f
        load[f] += sizes[c]
    fold = {n: cfold[camp[n]] for n in pos}
    neg = [n for n in names if n not in camp]
    for n, f in zip(neg, rng.integers(0, k, len(neg))):
        fold[n] = int(f)
    return fold


def event_percentiles(df, score="score"):
    """Per-event percentile and rank within the at-risk set of its month."""
    out = []
    for m, g in df.groupby("month", sort=False):
        ys = g["y"].to_numpy()
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        s = g[score].to_numpy()
        neg = np.sort(s[ys == 0])
        n = len(s)
        for i in np.where(ys == 1)[0]:
            below = np.searchsorted(neg, s[i], side="left")
            ties = np.searchsorted(neg, s[i], side="right") - below
            pct = (below + 0.5 * ties) / len(neg)
            # rank among the whole risk set, pessimistic on ties
            rank = (s > s[i]).sum() + ties + 1
            out.append((g["name"].iat[i], m, pct, rank / n, n))
    return pd.DataFrame(out, columns=["name", "month", "pct", "rank_frac", "risk_set"])


def summarize(ev, events=None, n_boot=1000, seed=0):
    """Point estimates + campaign-cluster bootstrap CIs."""
    res = {"events": len(ev), "auc": ev["pct"].mean()}
    for k in TOPK:
        res[f"recall@{k:.1%}"] = (ev["rank_frac"] <= k).mean()
    if events is not None and len(ev):
        camp = events.set_index("name")["campaign"]
        ev = ev.assign(campaign=ev["name"].map(camp))
        groups = [g for _, g in ev.groupby("campaign")]
        res["campaigns"] = len(groups)
        # Macro over campaigns: one worm wave does not dominate the number.
        res["auc_macro_campaign"] = np.mean([g["pct"].mean() for g in groups])
        res["recall@1%_macro_campaign"] = np.mean([(g["rank_frac"] <= 0.01).mean() for g in groups])
        rng = np.random.default_rng(seed)
        boots = {"auc": [], "recall@1.0%": []}
        for _ in range(n_boot):
            idx = rng.integers(0, len(groups), len(groups))
            b = pd.concat([groups[i] for i in idx])
            boots["auc"].append(b["pct"].mean())
            boots["recall@1.0%"].append((b["rank_frac"] <= 0.01).mean())
        for key, vals in boots.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            res[f"{key}_ci"] = (round(float(lo), 4), round(float(hi), 4))
    return res


def per_campaign(ev, events):
    camp = events.set_index("name")[["campaign", "t_event"]]
    ev = ev.join(camp, on="name")
    g = ev.groupby("campaign").agg(n=("pct", "size"), start=("t_event", "min"),
                                   auc=("pct", "mean"),
                                   recall1=("rank_frac", lambda r: (r <= 0.01).mean()),
                                   example=("name", "first"))
    return g.sort_values("n", ascending=False)
