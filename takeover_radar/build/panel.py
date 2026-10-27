"""
Build the monthly package-risk panel.

One row per (package, month) while the package is *at risk*: it has a
legitimate history (>= 3 versions, >= 90 days old) and has not yet been taken
over. Every feature is computed strictly from information published before the
first day of that month, so a row never sees its own outcome.

  y = 1  if the package's first takeover happens during that month.

Pass 1 scans every packument to build corpus-wide, time-indexed aggregates:
  * maintainer portfolio: how many packages list an account as maintainer
  * dependents (in-degree) within the corpus
  * when each maintainer account was first seen
Pass 2 computes per-package features against those aggregates.

Feature families (prefix):
  r_  registry: maintainers, publishers, publish hygiene (OIDC, provenance), cadence
  c_  corpus position: portfolio sizes, dependents, maintainer tenure
  g_  git contributor dynamics (the xz pattern): newcomers, maintainer fade,
      newcomers touching CI/build/release/fixtures
  x_  contagion: recently *reported* takeovers that share a maintainer or scope

Usage: python -m takeover_radar.build.panel --workers 12
"""

import argparse
import gzip
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from takeover_radar.collect.git_history import (BUILD, CI, FIXTURE, INSTALL, MANIFEST,
                                                normalize_repo, out_path)
from takeover_radar.paths import INTERIM, PACKUMENTS, PROCESSED, UNIVERSE, safe_name

EPOCH_YEAR = 2015
STUDY_START = (2017 - EPOCH_YEAR) * 12          # 2017-01
N_MONTHS = (2026 - EPOCH_YEAR) * 12 + 10        # through 2026-10 (inclusive)
MIN_PRIOR_VERSIONS = 3
MIN_AGE_DAYS = 90
SENSITIVE = CI | BUILD | INSTALL | FIXTURE | MANIFEST

WEBMAIL = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "msn.com",
    "yahoo.com", "ymail.com", "icloud.com", "me.com", "mac.com", "protonmail.com", "proton.me",
    "pm.me", "qq.com", "163.com", "126.com", "foxmail.com", "aol.com", "gmx.de", "gmx.net",
    "gmx.com", "mail.ru", "yandex.ru", "yandex.com", "web.de", "hey.com", "fastmail.com",
    "fastmail.fm", "zoho.com", "sina.com", "naver.com", "hotmail.co.uk", "yahoo.co.jp",
    "outlook.jp", "tutanota.com", "users.noreply.github.com", "npmjs.com", "github.com",
}


# ----------------------------------------------------------------------------- time helpers

def ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def month_of(t: datetime) -> int:
    return (t.year - EPOCH_YEAR) * 12 + t.month - 1


def month_start(m: int) -> datetime:
    return datetime(EPOCH_YEAR + m // 12, m % 12 + 1, 1)


MONTH_STARTS = [month_start(m) for m in range(N_MONTHS + 1)]
MONTH_EPOCH = np.array([d.timestamp() for d in MONTH_STARTS])


# Weekly panel: weeks start on Mondays; rows of never-taken-over weeks are sampled
# (hash of name and week) so the panel stays the size of the monthly one.
WEEK0 = datetime(2015, 1, 5)
N_WEEKS = (datetime(2026, 11, 1) - WEEK0).days // 7
WEEK_KEEP = 4          # keep 1 in WEEK_KEEP negative rows


def week_start(w: int) -> datetime:
    return WEEK0 + timedelta(days=7 * w)


def week_of(t: datetime) -> int:
    return (t - WEEK0).days // 7


def keep_week_row(name: str, w: int) -> bool:
    import zlib
    return zlib.crc32(f"{name}|{w}".encode()) % WEEK_KEEP == 0


def first_month_after(t: datetime) -> int:
    """Index of the first month whose start is strictly after t (pre-2015 history maps to 0)."""
    return max(0, min(month_of(t) + 1, N_MONTHS))


def load(name):
    p = PACKUMENTS / f"{safe_name(name)}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("missing"):
        return None
    vs = [v for v in doc.get("versions", []) if ts(v.get("t"))]
    for v in vs:
        v["_t"] = ts(v["t"])
    vs.sort(key=lambda v: v["_t"])
    doc["versions"] = vs
    return doc


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except (ValueError, TypeError):
        return -1


# ----------------------------------------------------------------------------- pass 1

def corpus_contrib(name):
    """Intervals [m0, m1) during which each maintainer / dependency is on the latest version."""
    doc = load(name)
    if doc is None or not doc["versions"]:
        return name, None
    vs = doc["versions"]
    maint_iv, dep_iv, first_seen = [], [], {}
    for i, v in enumerate(vs):
        m0 = first_month_after(v["_t"])
        m1 = first_month_after(vs[i + 1]["_t"]) if i + 1 < len(vs) else N_MONTHS
        for mt in v["maint"]:
            first_seen[mt] = min(first_seen.get(mt, v["_t"]), v["_t"])
        if m1 <= m0:
            continue
        maint_iv.append((m0, m1, tuple(v["maint"])))
        dep_iv.append((m0, m1, tuple(v["deps"])))
    return name, {"maint_iv": maint_iv, "dep_iv": dep_iv, "first_seen": first_seen}


def build_corpus(names, workers):
    maint_ids, pkg_ids = {}, {n: i for i, n in enumerate(names)}
    first_seen = {}
    port = defaultdict(lambda: np.zeros(N_MONTHS + 1, np.int32))
    indeg = np.zeros((len(names), N_MONTHS + 1), np.int32)
    with Pool(workers) as pool:
        for k, (name, c) in enumerate(pool.imap_unordered(corpus_contrib, names, chunksize=64), 1):
            if k % 10000 == 0:
                print(f"  corpus pass {k}/{len(names)}", flush=True)
            if c is None:
                continue
            for m0, m1, ms in c["maint_iv"]:
                for mt in ms:
                    a = port[mt]
                    a[m0] += 1
                    a[m1] -= 1
            for m0, m1, ds in c["dep_iv"]:
                for d in ds:
                    j = pkg_ids.get(d)
                    if j is not None:
                        indeg[j, m0] += 1
                        indeg[j, m1] -= 1
            for mt, t in c["first_seen"].items():
                if mt not in first_seen or t < first_seen[mt]:
                    first_seen[mt] = t
    for i, mt in enumerate(port):
        maint_ids[mt] = i
    port_arr = np.stack([np.cumsum(port[mt])[:N_MONTHS] for mt in maint_ids]) if maint_ids \
        else np.zeros((0, N_MONTHS), np.int32)
    indeg = np.cumsum(indeg, axis=1)[:, :N_MONTHS]
    return {"maint_ids": maint_ids, "pkg_ids": pkg_ids, "portfolio": port_arr.astype(np.int32),
            "indegree": indeg.astype(np.int32), "first_seen": first_seen}


# ----------------------------------------------------------------------------- pass 2

_C = {}


def _init(corpus_path, events_path, freq="month"):
    _C["freq"] = freq
    with open(corpus_path, "rb") as f:
        _C.update(pickle.load(f))
    ev = pd.read_json(events_path, lines=True)
    _C["events"] = {r.name_: r for r in ev.rename(columns={"name": "name_"}).itertuples()}
    # Contagion index: maintainer -> report times of takeovers of packages they maintained;
    # scope -> report times. Report time = when defenders could first know.
    by_maint, by_scope = defaultdict(list), defaultdict(list)
    for r in ev.itertuples():
        t_rep = ts(r.t_reported) or ts(r.t_event)
        for mt in _C["event_maint"].get(r.name, ()):
            by_maint[mt].append((t_rep, r.name))
        if r.name.startswith("@"):
            by_scope[r.name.split("/")[0]].append((t_rep, r.name))
    _C["by_maint"], _C["by_scope"] = by_maint, by_scope


def _git_arrays(repo):
    if not repo:
        return None
    p = out_path(repo)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        rec = json.load(f)
    return git_arrays_from_commits(rec.get("commits"))


def git_arrays_from_commits(cs):
    """Commit records [author_t, commit_t, author_id, committer_id, mask, nfiles, bot] -> arrays."""
    if not cs:
        return None
    a = np.array([[c[0], c[1], int(c[2], 16), int(c[3], 16), c[4], c[5], c[6]] for c in cs],
                 dtype=np.int64)
    a = a[a[:, 6] == 0]                     # drop bots
    a = a[(a[:, 0] > 946684800)]            # drop bogus pre-2000 timestamps
    if len(a) == 0:
        return None
    a = a[np.argsort(a[:, 0], kind="stable")]
    t = a[:, 0].astype(np.float64)
    author = a[:, 2]
    # first-commit time of each author, broadcast per commit
    uniq, inv = np.unique(author, return_inverse=True)
    first = np.full(len(uniq), np.inf)
    np.minimum.at(first, inv, t)
    return {"t": t, "author": inv, "committer_diff": (a[:, 2] != a[:, 3]),
            "mask": a[:, 4], "first": first[inv], "n_auth": len(uniq)}


def _git_features(g, T):
    """Contributor dynamics as of time T (epoch seconds)."""
    f = {}
    if g is None:
        f["g_available"] = 0.0
        return f
    day = 86400.0
    n = np.searchsorted(g["t"], T, side="left")
    f["g_available"] = 1.0
    if n == 0:
        f["g_commits_total"] = 0.0
        return f
    t, auth, mask, first = g["t"][:n], g["author"][:n], g["mask"][:n], g["first"][:n]
    f["g_commits_total"] = float(n)
    f["g_log_commits_total"] = math.log1p(n)
    f["g_days_since_commit"] = (T - t[-1]) / day
    f["g_repo_age_days"] = (T - t[0]) / day
    for w in (90, 365):
        sel = t >= T - w * day
        f[f"g_commits_{w}"] = float(sel.sum())
        f[f"g_authors_{w}"] = float(len(np.unique(auth[sel])))
    sel365 = t >= T - 365 * day
    if sel365.any():
        _, cnt = np.unique(auth[sel365], return_counts=True)
        cnt = np.sort(cnt)[::-1]
        f["g_top_share_365"] = cnt[0] / cnt.sum()
        f["g_bus_factor_365"] = float(np.searchsorted(np.cumsum(cnt), cnt.sum() / 2) + 1)
    # Maintainer fade: share of recent commits by the historically dominant author.
    hist = t < T - 365 * day
    if hist.sum() >= 10:
        a_h, c_h = np.unique(auth[hist], return_counts=True)
        top = a_h[np.argmax(c_h)]
        f["g_hist_top_share"] = c_h.max() / c_h.sum()
        rec = t >= T - 180 * day
        f["g_hist_top_recent_share"] = float((auth[rec] == top).mean()) if rec.any() else 0.0
        f["g_hist_top_days_since"] = (T - t[auth == top][-1]) / day
    # Newcomers: authors whose first commit is within the window.
    for w in (180, 365):
        rec = t >= T - w * day
        newc = rec & (first >= T - w * day)
        f[f"g_newcomer_commits_{w}"] = float(newc.sum())
        f[f"g_newcomer_share_{w}"] = float(newc.sum() / max(rec.sum(), 1))
        f[f"g_newcomer_sensitive_{w}"] = float((newc & ((mask & SENSITIVE) > 0)).sum())
        f[f"g_newcomer_fixture_{w}"] = float((newc & ((mask & FIXTURE) > 0)).sum())
        if newc.any():
            _, cn = np.unique(auth[newc], return_counts=True)
            f[f"g_newcomer_max_{w}"] = float(cn.max())
            f[f"g_newcomers_{w}"] = float(len(cn))
        else:
            f[f"g_newcomer_max_{w}"] = 0.0
            f[f"g_newcomers_{w}"] = 0.0
    rec = t >= T - 180 * day
    f["g_sensitive_180"] = float(((mask[rec] & SENSITIVE) > 0).sum())
    f["g_ci_180"] = float(((mask[rec] & CI) > 0).sum())
    f["g_fixture_180"] = float(((mask[rec] & FIXTURE) > 0).sum())
    f["g_applied_by_other_180"] = float(g["committer_diff"][:n][rec].mean()) if rec.any() else 0.0
    # Dormant-then-revived repository
    if n >= 2:
        gaps = np.diff(t) / day
        recent_gap_idx = np.where(t[1:] >= T - 180 * day)[0]
        f["g_revival_gap_days"] = float(gaps[recent_gap_idx].max()) if len(recent_gap_idx) else 0.0
    return f


def _registry_features(name, vs, T, m, C):
    f = {}
    k = C["pkg_ids"].get(name)
    if k is not None:
        deg = C["indegree"][k]
        f["c_indegree"] = math.log1p(max(deg[m], 0))
        f["c_indegree_growth_12"] = math.log1p(max(deg[m], 0)) - math.log1p(max(deg[max(m - 12, 0)], 0))
    day = 86400.0
    Te = T.timestamp()
    last = vs[-1]
    times = np.array([v["_t"].timestamp() for v in vs])
    f["r_n_versions"] = float(len(vs))
    f["r_log_n_versions"] = math.log1p(len(vs))
    f["r_age_days"] = (Te - times[0]) / day
    f["r_days_since_pub"] = (Te - times[-1]) / day
    for w in (90, 365):
        f[f"r_pubs_{w}"] = float((times >= Te - w * day).sum())
    if len(times) >= 3:
        gaps = np.diff(times) / day
        f["r_gap_cv"] = float(gaps.std() / (gaps.mean() + 1e-6))
        f["r_last_gap_days"] = float(gaps[-1])
        f["r_last_gap_ratio"] = float(gaps[-1] / (np.median(gaps) + 1e-3))
    # Maintainers & publishers
    maint = last["maint"]
    f["r_n_maint"] = float(len(maint))
    ever_m, ever_p = set(), set()
    seen_pub = set()
    added = {90: 0, 365: 0}
    removed = {90: 0, 365: 0}
    newpub = {90: 0, 365: 0}
    prev = None
    for v, tv in zip(vs, times):
        ever_m.update(v["maint"])
        if v.get("pub"):
            if v["pub"] not in seen_pub and seen_pub:
                for w in (90, 365):
                    if tv >= Te - w * day:
                        newpub[w] += 1
            seen_pub.add(v["pub"])
        if prev is not None:
            a = set(v["maint"]) - set(prev["maint"])
            r = set(prev["maint"]) - set(v["maint"])
            for w in (90, 365):
                if tv >= Te - w * day:
                    added[w] += len(a)
                    removed[w] += len(r)
        prev = v
    pubs = [v.get("pub") for v in vs if v.get("pub")]
    f["r_n_maint_ever"] = float(len(ever_m))
    f["r_n_publishers_ever"] = float(len(set(pubs)))
    f["r_n_publishers_365"] = float(len({v.get("pub") for v, tv in zip(vs, times)
                                         if tv >= Te - 365 * day and v.get("pub")}))
    if pubs:
        _, cnt = np.unique(pubs, return_counts=True)
        f["r_top_pub_share"] = cnt.max() / cnt.sum()
        f["r_last_pub_is_new"] = float(pubs.count(pubs[-1]) == 1 and len(pubs) > 1)
        f["r_last_pub_in_maint"] = float(pubs[-1] in maint)
    for w in (90, 365):
        f[f"r_maint_added_{w}"] = float(added[w])
        f[f"r_maint_removed_{w}"] = float(removed[w])
        f[f"r_new_publisher_{w}"] = float(newpub[w])
    # Publish hygiene over the last five releases
    tail = vs[-5:]
    f["r_tp_share5"] = float(np.mean([bool(v.get("tp")) for v in tail]))
    f["r_att_share5"] = float(np.mean([bool(v.get("att")) for v in tail]))
    f["r_githead_share5"] = float(np.mean([bool(v.get("gh")) for v in tail]))
    f["r_last_tp"] = float(bool(last.get("tp")))
    f["r_last_att"] = float(bool(last.get("att")))
    f["r_ever_tp"] = float(any(v.get("tp") for v in vs))
    f["r_ever_att"] = float(any(v.get("att") for v in vs))
    f["r_att_dropped"] = float(f["r_ever_att"] and not last.get("att"))
    f["r_npm_major"] = float(_major(last.get("npmv")))
    f["r_node_major"] = float(_major(last.get("nodev")))
    if len(vs) >= 2:
        f["r_tool_change"] = float(_major(vs[-2].get("npmv")) != _major(last.get("npmv")))
    f["r_hooks"] = float(bool(last.get("hooks")))
    f["r_ndeps"] = float(last.get("ndeps") or 0)
    f["r_log_size"] = math.log1p(last.get("size") or 0)
    f["r_files"] = math.log1p(last.get("files") or 0)
    f["r_deprecated"] = float(bool(last.get("depr")))
    f["r_has_repo"] = float(bool(last.get("repo")))
    doms = last.get("maint_dom") or []
    custom = [d for d in doms if d not in WEBMAIL]
    f["r_custom_domain_share"] = len(custom) / len(doms) if doms else 0.0
    f["r_any_custom_domain"] = float(bool(custom))
    # Corpus position
    ids = [C["maint_ids"][mt] for mt in maint if mt in C["maint_ids"]]
    if ids:
        port = C["portfolio"][ids, m]
        f["c_portfolio_max"] = math.log1p(port.max())
        f["c_portfolio_sum"] = math.log1p(port.sum())
        f["c_portfolio_min"] = math.log1p(port.min())
    pid = C["maint_ids"].get(last.get("pub"))
    if pid is not None:
        f["c_publisher_portfolio"] = math.log1p(C["portfolio"][pid, m])
    tenures = [(T - C["first_seen"][mt]).days for mt in maint if mt in C["first_seen"]]
    if tenures:
        f["c_maint_tenure_min"] = float(min(tenures))
        f["c_maint_tenure_max"] = float(max(tenures))
    return f


def _contagion(name, maint, T, C):
    f = {}
    for w in (30, 90):
        lo = T - timedelta(days=w)
        hits = set()
        for mt in maint:
            for t_rep, other in C["by_maint"].get(mt, ()):
                if lo <= t_rep < T and other != name:
                    hits.add(other)
        f[f"x_comaint_reported_{w}"] = float(len(hits))
        if name.startswith("@"):
            sc = sum(1 for t_rep, other in C["by_scope"].get(name.split("/")[0], ())
                     if lo <= t_rep < T and other != name)
            f[f"x_scope_reported_{w}"] = float(sc)
    return f


def package_rows(name):
    doc = load(name)
    if doc is None:
        return []
    repo = None
    if doc["versions"]:
        repo = normalize_repo(doc["versions"][-1].get("repo") or doc.get("repo"))
    fn = rows_from_weekly if _C.get("freq") == "week" else rows_from
    return fn(name, doc, _git_arrays(repo))


def rows_from(name, doc, g, months=None):
    """Panel rows for one package from an in-memory packument and git arrays.

    `months` restricts output to specific month indices (used by the synthetic
    stress test, which edits histories in memory)."""
    C = _C
    vs = doc["versions"]
    if len(vs) < MIN_PRIOR_VERSIONS:
        return []
    ev = C["events"].get(name)
    t_event = ts(ev.t_event) if ev is not None else None
    mal = set(ev.malicious_versions) if ev is not None else set()
    vs = [v for v in vs if v["v"] not in mal and (t_event is None or v["_t"] < t_event)]
    if len(vs) < MIN_PRIOR_VERSIONS:
        return []
    start_t = max(vs[MIN_PRIOR_VERSIONS - 1]["_t"], vs[0]["_t"] + timedelta(days=MIN_AGE_DAYS))
    m0 = max(first_month_after(start_t), STUDY_START)
    m_event = month_of(t_event) if t_event else None
    m1 = min(m_event + 1 if m_event is not None else N_MONTHS, N_MONTHS)
    if m0 >= m1:
        return []
    vtimes = [v["_t"] for v in vs]
    rows = []
    j = 0
    for m in range(m0, m1):
        if months is not None and m not in months:
            continue
        T = MONTH_STARTS[m]
        while j < len(vs) and vtimes[j] < T:
            j += 1
        hist = vs[:j]
        if len(hist) < MIN_PRIOR_VERSIONS:
            continue
        row = {"name": name, "month": m, "y": int(m_event == m)}
        row.update(features_at(name, hist, g, m))
        rows.append(row)
    return rows


def rows_from_weekly(name, doc, g):
    """Weekly rows: same risk set and features as rows_from, evaluated at each Monday.

    Corpus aggregates (portfolio, dependents) stay monthly: the value for the
    month containing the week start is computed from data before that month."""
    C = _C
    vs = doc["versions"]
    if len(vs) < MIN_PRIOR_VERSIONS:
        return []
    ev = C["events"].get(name)
    t_event = ts(ev.t_event) if ev is not None else None
    mal = set(ev.malicious_versions) if ev is not None else set()
    vs = [v for v in vs if v["v"] not in mal and (t_event is None or v["_t"] < t_event)]
    if len(vs) < MIN_PRIOR_VERSIONS:
        return []
    start_t = max(vs[MIN_PRIOR_VERSIONS - 1]["_t"], vs[0]["_t"] + timedelta(days=MIN_AGE_DAYS))
    w0 = max(week_of(start_t) + 1, week_of(MONTH_STARTS[STUDY_START]) + 1)
    w_event = week_of(t_event) if t_event else None
    w1 = min(w_event + 1 if w_event is not None else N_WEEKS, N_WEEKS)
    vtimes = [v["_t"] for v in vs]
    rows, j = [], 0
    for w in range(w0, w1):
        y = int(w_event == w)
        if not y and not keep_week_row(name, w):
            continue
        T = week_start(w)
        while j < len(vs) and vtimes[j] < T:
            j += 1
        if j < MIN_PRIOR_VERSIONS:
            continue
        m = month_of(T)
        row = {"name": name, "week": w, "month": m, "y": y}
        row.update(features_at(name, vs[:j], g, m, T))
        rows.append(row)
    return rows


def features_at(name, hist, g, m, T=None):
    """All features for one package given its legitimate history before time T.

    T defaults to the first day of month m; live scoring passes the current time
    and the latest month, whose corpus aggregates are used."""
    C = _C
    T = T or MONTH_STARTS[m]
    f = _registry_features(name, hist, T, m, C)
    f.update(_git_features(g, T.timestamp()))
    f.update(_contagion(name, hist[-1]["maint"], T, C))
    f["r_scoped"] = float(name.startswith("@"))
    return f


# ----------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N packages")
    ap.add_argument("--rebuild-corpus", action="store_true")
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--freq", choices=["month", "week"], default="month")
    ap.add_argument("--merge-only", action="store_true",
                    help="only merge existing part files (after a crash in the final merge)")
    args = ap.parse_args()

    with open(UNIVERSE, encoding="utf-8") as f:
        universe = list(dict.fromkeys(json.loads(l)["name"] for l in f))
    events = pd.read_json(INTERIM / "events.jsonl", lines=True)
    names = list(dict.fromkeys(universe + events["name"].tolist()))
    if args.limit:
        keep = set(names[:args.limit]) | set(events["name"])
        names = [n for n in names if n in keep]
    print(f"{len(names)} packages ({len(universe)} universe, {len(events)} events)", flush=True)

    corpus_path = INTERIM / ("corpus_debug.pkl" if args.limit else "corpus.pkl")
    if corpus_path.exists() and not args.limit and not args.rebuild_corpus:
        print("reusing corpus aggregates", flush=True)
    else:
        # Aggregates come from the universe only. Adding out-of-universe takeover victims
        # would inflate portfolio/dependents for exactly the maintainers that got hit.
        uni_set = set(universe)
        corpus = build_corpus([n for n in names if n in uni_set], args.workers)
        # maintainer set of each taken-over package, for contagion features
        event_maint = {}
        for n in events["name"]:
            doc = load(n)
            if doc and doc["versions"]:
                t_e = ts(events.set_index("name").loc[n, "t_event"])
                prior = [v for v in doc["versions"] if v["_t"] < t_e]
                if prior:
                    event_maint[n] = tuple(prior[-1]["maint"])
        corpus["event_maint"] = event_maint
        with open(corpus_path, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"corpus: {len(corpus['maint_ids'])} maintainers", flush=True)
    if args.corpus_only:
        return

    parts, buf, total = [], [], 0
    stem = ("panel_debug" if args.limit else "panel") + ("_weekly" if args.freq == "week" else "")
    out_dir = PROCESSED / stem
    if args.merge_only:
        merge_parts(sorted(out_dir.glob("part-*.parquet")), set(universe), PROCESSED / f"{stem}.parquet")
        return
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    with Pool(args.workers, initializer=_init,
              initargs=(str(corpus_path), str(INTERIM / "events.jsonl"), args.freq)) as pool:
        for k, rows in enumerate(pool.imap_unordered(package_rows, names, chunksize=32), 1):
            buf.extend(rows)
            if len(buf) >= 400_000 or k == len(names):
                df = pd.DataFrame(buf)
                # float32 per part keeps the final concat at half the memory
                feat = [c for c in df.columns if c[:2] in ("r_", "c_", "g_", "x_")]
                df[feat] = df[feat].astype(np.float32)
                p = out_dir / f"part-{len(parts):04d}.parquet"
                df.to_parquet(p, index=False)
                parts.append(p)
                total += len(buf)
                buf = []
            if k % 5000 == 0 or k == len(names):
                print(f"  panel {k}/{len(names)} rows={total + len(buf)}", flush=True)
    merge_parts(parts, set(universe), PROCESSED / f"{stem}.parquet")


def merge_parts(parts, uni, out):
    """Stream the part files into one parquet, one part in memory at a time.

    A full pd.concat of the ~7M-row panel ran the machine out of memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, events, packages, writer = 0, 0, set(), None
    for p in parts:
        df = pd.read_parquet(p)
        df["in_universe"] = df["name"].isin(uni).astype(np.int8)
        feat = [c for c in df.columns if c[:2] in ("r_", "c_", "g_", "x_")]
        df[feat] = df[feat].astype(np.float32)
        if writer is None:
            cols = list(df.columns)
            table = pa.Table.from_pandas(df[cols], preserve_index=False)
            schema = table.schema.remove_metadata()
            writer = pq.ParquetWriter(out, schema)
        else:
            table = pa.Table.from_pandas(df[cols], preserve_index=False)
        writer.write_table(table.cast(schema))
        rows += len(df)
        events += int(df["y"].sum())
        packages.update(df["name"].unique())
        del df, table
    writer.close()
    for p in parts:
        p.unlink()
    print(f"panel: {rows} rows, {len(packages)} packages, {events} events, "
          f"{len(feat)} features", flush=True)


if __name__ == "__main__":
    main()
