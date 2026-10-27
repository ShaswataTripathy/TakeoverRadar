"""
Fetch the study universe: the most-downloaded npm packages from ecosyste.ms.

Only used to choose *which* packages to study and to stratify evaluation by
popularity. Current download counts are never used as a model feature,
because they are measured after the incidents we try to forecast.

Usage: python -m takeover_radar.collect.universe --pages 100
"""

import argparse
import json
import time

import requests

from takeover_radar.paths import UNIVERSE

API = "https://packages.ecosyste.ms/api/v1/registries/npmjs.org/packages"
KEEP = ("name", "downloads", "dependent_packages_count", "dependent_repos_count",
        "repository_url", "versions_count", "first_release_published_at",
        "latest_release_published_at", "status")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument("--per-page", type=int, default=1000)
    args = ap.parse_args()

    done = set()
    if UNIVERSE.exists():
        with open(UNIVERSE, encoding="utf-8") as f:
            done = {json.loads(l)["_page"] for l in f}

    with open(UNIVERSE, "a", encoding="utf-8") as out:
        for page in range(1, args.pages + 1):
            if page in done:
                continue
            for attempt in range(6):
                try:
                    r = requests.get(API, params={"sort": "downloads", "order": "desc",
                                                  "per_page": args.per_page, "page": page},
                                     timeout=120)
                    r.raise_for_status()
                    rows = r.json()
                    break
                except Exception as e:  # network hiccups / 5xx
                    print(f"page {page} attempt {attempt}: {e}", flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                raise SystemExit(f"giving up on page {page}")
            for p in rows:
                rec = {k: p.get(k) for k in KEEP}
                rec["_page"] = page
                out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"page {page}/{args.pages}: {len(rows)} rows, last downloads="
                  f"{rows[-1].get('downloads') if rows else None}", flush=True)


if __name__ == "__main__":
    main()
