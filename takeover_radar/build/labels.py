"""
Turn malicious-version reports into takeover labels.

A package counts as a *takeover* (event) when a version reported malicious
was published after the package already had a legitimate history:

  * at least MIN_PRIOR_VERSIONS versions published before the first malicious one,
  * the first legitimate version is at least MIN_PRIOR_DAYS older,
  * at least one prior version is still live on the registry
    (the registry did not wipe the package as attacker-owned).

Everything else in the reports (typosquats, dependency confusion, spam,
packages malicious from their first release) is excluded from the event set
and from the at-risk population, since those were never legitimate.

Known maintainer self-sabotage (protestware) is excluded: the author was not
taken over, so it is a different phenomenon.

Events are grouped into campaigns by single-linkage on event time (gap
CAMPAIGN_GAP_DAYS) so evaluation can hold out whole campaigns at once; worm
waves would otherwise leak between train and test.

Usage: python -m takeover_radar.build.labels
"""

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from takeover_radar.paths import INTERIM, PACKUMENTS, safe_name

MIN_PRIOR_VERSIONS = 3
MIN_PRIOR_DAYS = 90
CAMPAIGN_GAP_DAYS = 3

# Author-published sabotage, not account takeover.
PROTESTWARE = {"node-ipc", "colors", "faker", "peacenotwar", "es5-ext", "event-source-polyfill",
               "styled-components-native", "@medicalwebexperts/react-native-lightbox"}


def parse_ts(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def load_packument(name):
    p = PACKUMENTS / f"{safe_name(name)}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def malicious_versions(reports, doc):
    """Resolve the set of reported versions present in the packument time map."""
    times = doc.get("time") or {}
    vers = set()
    for r in reports:
        vers.update(v for v in r["versions"] if v in times)
        # Range-only advisories (some GHSA): take the explicitly introduced versions.
        vers.update(v for v in r["introduced"] if v != "0" and v in times)
    return vers


def build():
    reports = defaultdict(list)
    with open(INTERIM / "osv_reports.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["whole_package"]:
                reports[r["name"]].append(r)

    events, reasons = [], Counter()
    for name, reps in reports.items():
        if name in PROTESTWARE:
            reasons["protestware"] += 1
            continue
        doc = load_packument(name)
        if doc is None:
            reasons["not_fetched"] += 1
            continue
        if doc.get("missing") or doc.get("fully_unpublished"):
            reasons["gone_from_registry"] += 1
            continue
        mal = malicious_versions(reps, doc)
        if not mal:
            reasons["versions_not_found"] += 1
            continue
        times = doc["time"]
        t_mal = min(parse_ts(times[v]) for v in mal)
        live = {v["v"] for v in doc["versions"]}
        prior = [(parse_ts(t), v) for v, t in times.items()
                 if v not in mal and parse_ts(t) and parse_ts(t) < t_mal]
        prior_live = [x for x in prior if x[1] in live]
        if len(prior) < MIN_PRIOR_VERSIONS:
            reasons["short_prior_history"] += 1
            continue
        if min(t for t, _ in prior) > t_mal - timedelta(days=MIN_PRIOR_DAYS):
            reasons["young_package"] += 1
            continue
        if not prior_live:
            reasons["no_live_prior_version"] += 1
            continue
        reasons["takeover"] += 1
        events.append({
            "name": name,
            "t_event": t_mal.isoformat(),
            "malicious_versions": sorted(mal),
            "n_prior_versions": len(prior),
            "first_publish": min(t for t, _ in prior).isoformat(),
            "report_ids": sorted({r["id"] for r in reps}),
            # Earliest public report: when a defender could first have known.
            "t_reported": min(r["published"] for r in reps if r.get("published")),
            "summary": reps[0]["summary"],
        })

    # Campaigns: single linkage over sorted event times.
    events.sort(key=lambda e: e["t_event"])
    cid, last = -1, None
    for e in events:
        t = parse_ts(e["t_event"])
        if last is None or t - last > timedelta(days=CAMPAIGN_GAP_DAYS):
            cid += 1
        e["campaign"] = cid
        last = t
    sizes = Counter(e["campaign"] for e in events)
    for e in events:
        e["campaign_size"] = sizes[e["campaign"]]

    with open(INTERIM / "events.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print("filter outcome:", dict(reasons))
    print(f"{len(events)} takeover events in {len(sizes)} campaigns")
    by_c = defaultdict(list)
    for e in events:
        by_c[e["campaign"]].append(e)
    for c, n in sizes.most_common(25):
        es = by_c[c]
        print(f"  campaign {c:4d} n={n:4d} {es[0]['t_event'][:10]}..{es[-1]['t_event'][:10]} "
              f"e.g. {', '.join(x['name'] for x in es[:4])}")


if __name__ == "__main__":
    build()
