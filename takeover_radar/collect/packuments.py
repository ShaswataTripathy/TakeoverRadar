"""
Fetch full npm packuments and keep only the fields needed to rebuild a
package's publishing and maintainer history.

The full packument records, for every version: the account that published it
(`_npmUser`, including trusted-publisher / OIDC info), the maintainer list *as
it was at publish time*, tooling versions, gitHead, provenance attestations and
dependencies. The `time` map keeps timestamps even for versions that were
later unpublished, which is how malicious releases are dated.

Usage: python -m takeover_radar.collect.packuments --names data/interim/names.txt
"""

import argparse
import asyncio
import gzip
import json
import random
import time
from urllib.parse import quote

import aiohttp

from takeover_radar.paths import PACKUMENTS, safe_name

REGISTRY = "https://registry.npmjs.org/"
INSTALL_HOOKS = ("preinstall", "install", "postinstall", "prepare")


def _domain(email):
    if not email or "@" not in email:
        return None
    return email.rsplit("@", 1)[1].lower()


def _repo_url(repo):
    if isinstance(repo, dict):
        repo = repo.get("url")
    return repo if isinstance(repo, str) else None


def compact(doc: dict) -> dict:
    times = {k: v for k, v in (doc.get("time") or {}).items()
             if k not in ("created", "modified")}
    versions = doc.get("versions") or {}
    out_versions = []
    for ver, v in versions.items():
        if not isinstance(v, dict):
            continue
        user = v.get("_npmUser") or {}
        if isinstance(user, str):  # very old packuments: "name <email>"
            user = {"name": user.split(" ")[0]}
        maint = [m for m in (v.get("maintainers") or []) if isinstance(m, dict)]
        dist = v.get("dist") or {}
        scripts = v.get("scripts") if isinstance(v.get("scripts"), dict) else {}
        deps = v.get("dependencies") if isinstance(v.get("dependencies"), dict) else {}
        out_versions.append({
            "v": ver,
            "t": times.get(ver),
            "pub": user.get("name"),
            "pub_dom": _domain(user.get("email")),
            "tp": (user.get("trustedPublisher") or {}).get("id"),
            "appr": (user.get("approver") or {}).get("name"),
            "maint": sorted({m.get("name") for m in maint if m.get("name")}),
            "maint_dom": sorted({d for d in (_domain(m.get("email")) for m in maint) if d}),
            "gh": bool(v.get("gitHead")),
            "npmv": v.get("_npmVersion"),
            "nodev": v.get("_nodeVersion"),
            "att": "attestations" in dist,
            "nsig": "npm-signature" in dist,
            "hooks": [h for h in INSTALL_HOOKS if h in scripts],
            "deps": sorted(deps)[:400],
            "ndeps": len(deps),
            "files": dist.get("fileCount"),
            "size": dist.get("unpackedSize"),
            "repo": _repo_url(v.get("repository")),
            "depr": bool(v.get("deprecated")),
        })
    out_versions.sort(key=lambda x: x["t"] or "")
    return {
        "name": doc.get("name"),
        "created": (doc.get("time") or {}).get("created"),
        "modified": (doc.get("time") or {}).get("modified"),
        "repo": _repo_url(doc.get("repository")),
        "time": times,
        "unpublished": sorted(k for k in times if k not in versions),
        "tags": doc.get("dist-tags") or {},
        "versions": out_versions,
        "fully_unpublished": bool((doc.get("time") or {}).get("unpublished")),
    }


async def fetch_one(session, sem, name, stats):
    path = PACKUMENTS / f"{safe_name(name)}.json.gz"
    if path.exists():
        stats["skip"] += 1
        return
    url = REGISTRY + quote(name, safe="@")
    async with sem:
        for attempt in range(6):
            try:
                async with session.get(url, headers={"Accept": "application/json"}) as r:
                    if r.status == 404:
                        rec = {"name": name, "missing": True}
                        break
                    if r.status in (429, 500, 502, 503, 504):
                        raise aiohttp.ClientResponseError(r.request_info, (), status=r.status)
                    r.raise_for_status()
                    doc = json.loads(await r.read())
                    rec = compact(doc)
                    break
            except Exception as e:
                if attempt == 5:
                    stats["fail"] += 1
                    print(f"FAIL {name}: {e!r}", flush=True)
                    return
                await asyncio.sleep(2 ** attempt + random.random())
    # compaction of huge packuments is CPU work; write atomically
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
    tmp.replace(path)
    stats["missing" if rec.get("missing") else "ok"] += 1


async def run(names, concurrency):
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "skip": 0, "missing": 0, "fail": 0}
    timeout = aiohttp.ClientTimeout(total=300)
    conn = aiohttp.TCPConnector(limit=concurrency)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(fetch_one(session, sem, n, stats)) for n in names]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            await fut
            if i % 500 == 0 or i == len(tasks):
                rate = i / (time.time() - t0)
                print(f"{i}/{len(tasks)} {stats} {rate:.1f}/s "
                      f"eta {(len(tasks) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    with open(args.names, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    names = list(dict.fromkeys(names))
    print(f"{len(names)} packages", flush=True)
    asyncio.run(run(names, args.concurrency))


if __name__ == "__main__":
    main()
