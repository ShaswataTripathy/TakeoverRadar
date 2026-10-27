"""
Score npm packages for maintainer-takeover risk right now.

Fetches the live packument and the repository's commit history, computes the
same features the model was trained on (including the pretrained-sequence
features over the package's monthly history), and prints the package's risk
percentile among ~120k popular npm packages plus the signals that drove it.

The percentile is the deployed blend: the mean of the package's percentile by
popularity (dependents) and by the GBM, each against the reference month.

Usage:
  python score.py axios left-pad @ctrl/tinycolor
  python score.py --file names.txt --json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from urllib.parse import quote

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from takeover_radar.build import panel as P
from takeover_radar.collect.git_history import extract, normalize_repo
from takeover_radar.collect.packuments import REGISTRY, compact
from takeover_radar.models.gbm import popularity_score
from takeover_radar.models.pretrain import load_encoder, seq_features
from takeover_radar.paths import INTERIM, MODEL_DIR

DESCRIBE = {
    "r_n_maint": "number of maintainers",
    "c_indegree": "dependents within popular npm",
    "c_portfolio_max": "largest maintainer portfolio (packages one account controls)",
    "c_portfolio_sum": "total packages controlled by the maintainers",
    "c_publisher_portfolio": "packages controlled by the last publisher",
    "c_maint_tenure_min": "age of the newest maintainer account (days)",
    "r_last_tp": "last release used trusted publishing (OIDC)",
    "r_tp_share5": "share of last 5 releases via trusted publishing",
    "r_att_share5": "share of last 5 releases with provenance",
    "r_ever_att": "ever published with provenance",
    "r_days_since_pub": "days since last release",
    "r_pubs_365": "releases in the last year",
    "r_maint_added_365": "maintainers added in the last year",
    "r_new_publisher_365": "new publishing accounts in the last year",
    "r_any_custom_domain": "a maintainer uses a custom email domain",
    "g_newcomer_share_365": "share of recent commits by newcomers",
    "g_newcomer_sensitive_365": "newcomer commits touching CI/build/release files",
    "g_hist_top_recent_share": "recent activity of the long-time top contributor",
    "g_bus_factor_365": "contributors behind half of last year's commits",
    "x_comaint_reported_30": "co-maintained packages reported compromised (30d)",
    "x_scope_reported_30": "same-scope packages reported compromised (30d)",
    "s_surprise": "how unusual this month is for the package's own history",
    "s_surprise_registry": "unusual registry activity vs the package's history",
    "s_surprise_git": "unusual contributor activity vs the package's history",
}


def fetch_doc(name):
    r = requests.get(REGISTRY + quote(name, safe="@"), headers={"Accept": "application/json"}, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    doc = compact(r.json())
    vs = [v for v in doc["versions"] if P.ts(v.get("t"))]
    for v in vs:
        v["_t"] = P.ts(v["t"])
    doc["versions"] = sorted(vs, key=lambda v: v["_t"])
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    names = list(args.names)
    if args.file:
        names += [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    if not names:
        ap.error("give package names or --file")

    model = lgb.Booster(model_file=str(MODEL_DIR / "takeover_radar_gbm.txt"))
    feats = model.feature_name()
    ref = np.load(MODEL_DIR / "reference_scores.npy")
    ref_pop = np.load(MODEL_DIR / "reference_popularity.npy")
    encoder, meta = load_encoder()
    P._init(str(INTERIM / "corpus.pkl"), str(INTERIM / "events.jsonl"))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    m = P.N_MONTHS - 1

    out = []
    for name in names:
        doc = fetch_doc(name)
        if doc is None or len(doc["versions"]) < P.MIN_PRIOR_VERSIONS:
            out.append({"name": name, "error": "not found or fewer than 3 versions"})
            continue
        g = None
        repo = normalize_repo(doc["versions"][-1].get("repo") or doc.get("repo"))
        if repo and not args.no_git:
            rec = extract(repo, timeout=600)
            g = P.git_arrays_from_commits(rec.get("commits"))
        f = P.features_at(name, doc["versions"], g, m, T=now)
        # monthly history up to last month plus the current state, for the sequence encoder
        hist = P.rows_from(name, doc, g, months=set(range(P.STUDY_START, m)))
        rows = pd.DataFrame(hist + [dict(f, name=name, month=m, y=0)])
        f.update(seq_features(rows, encoder, meta).iloc[-1].drop(["name", "month"]).to_dict())
        x = np.array([[f.get(c, np.nan) for c in feats]], np.float32)
        raw = float(model.predict(x, raw_score=True)[0])
        contrib = model.predict(x, pred_contrib=True)[0][:-1]
        pop = float(popularity_score(pd.DataFrame([f]).reindex(
            columns=["c_indegree", "r_log_n_versions"])).iloc[0])
        pct = 0.5 * (np.searchsorted(ref, raw) / len(ref) + np.searchsorted(ref_pop, pop) / len(ref_pop))
        top = sorted(zip(feats, contrib, x[0]), key=lambda t: -abs(t[1]))[:args.top]
        out.append({
            "name": name, "risk_percentile": round(100 * pct, 2), "repo": repo,
            "drivers": [{"feature": c, "meaning": DESCRIBE.get(c, c),
                         "value": None if math.isnan(v) else round(float(v), 3),
                         "effect": round(float(e), 3)} for c, e, v in top],
        })

    if args.json:
        json.dump(out, sys.stdout, indent=1)
        return
    for o in out:
        if "error" in o:
            print(f"{o['name']}: {o['error']}")
            continue
        band = "HIGH" if o["risk_percentile"] >= 99 else "elevated" if o["risk_percentile"] >= 95 else "typical"
        print(f"\n{o['name']}  risk percentile {o['risk_percentile']:.1f}  ({band})")
        for d in o["drivers"]:
            arrow = "raises" if d["effect"] > 0 else "lowers"
            print(f"   {arrow:6s} {d['meaning']} = {d['value']}  ({d['effect']:+.2f})")


if __name__ == "__main__":
    main()
