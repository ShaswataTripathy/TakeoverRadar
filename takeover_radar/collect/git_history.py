"""
Extract contributor history from each package's source repository.

Uses a blobless bare clone (commits + trees, no file contents), so even large
repositories are cheap. For every commit we keep author/committer time, a
salted hash of the author identity and a bitmask of which kinds of files it
touched. The clone is deleted right after extraction.

These are the signals behind slow social takeovers (the xz pattern): a new
contributor ramps up, the long-time maintainer fades, and the newcomer starts
touching CI, build, release and binary test-fixture files.

Usage: python -m takeover_radar.collect.git_history --repos data/interim/repos.txt
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from takeover_radar.paths import GIT_LOGS

SALT = b"takeover-radar-v1"
BOT = re.compile(r"\[bot\]|dependabot|renovate|greenkeeper|github-actions|semantic-release|"
                 r"snyk-bot|noreply@github\.com$|action@github\.com", re.I)

# File-kind bits
CI, MANIFEST, BUILD, FIXTURE, SRC, TEST, DOCS, LOCK, INSTALL = (1 << i for i in range(9))
CODE_EXT = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".coffee", ".json", ".vue", ".svelte")
DOC_EXT = (".md", ".markdown", ".txt", ".rst", ".html")


def file_kind(path: str) -> int:
    p = path.lower()
    base = p.rsplit("/", 1)[-1]
    if p.startswith((".github/workflows/", ".circleci/", ".gitlab-ci", ".travis", "azure-pipelines",
                     ".buildkite/", "appveyor", "jenkinsfile")) or base in (".travis.yml", "appveyor.yml"):
        return CI
    if base == "package.json":
        return MANIFEST
    if base in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "bun.lockb"):
        return LOCK
    if base in (".npmignore", ".npmrc", ".releaserc", ".releaserc.json", "release.config.js",
                "makefile", "gulpfile.js", "gruntfile.js", "rollup.config.js", "webpack.config.js",
                "tsup.config.ts", "vite.config.ts", "babel.config.js", ".babelrc", "lerna.json",
                "nx.json", "turbo.json", "configure.ac", "cmakelists.txt", "binding.gyp") \
            or base.startswith(("build.", "rollup.", "webpack.", "esbuild.")):
        return BUILD
    if base in ("postinstall.js", "preinstall.js", "install.js", "setup.js", "setup.mjs") \
            or p.startswith("scripts/"):
        return INSTALL
    in_test = re.search(r"(^|/)(tests?|__tests__|spec|fixtures?)(/|$)", p) is not None
    if in_test and not p.endswith(CODE_EXT + DOC_EXT):
        return FIXTURE  # binary / opaque test data: where xz hid its payload
    if in_test or ".test." in base or ".spec." in base:
        return TEST
    if p.endswith(DOC_EXT) or p.startswith(("docs/", "doc/", "examples/")):
        return DOCS
    return SRC


def ident(email: str, name: str) -> str:
    key = (email or name or "").strip().lower().encode()
    return hashlib.sha1(SALT + key).hexdigest()[:12]


def normalize_repo(url):
    """Return 'host/owner/name' for GitHub/GitLab/Bitbucket URLs, else None."""
    if not url:
        return None
    u = url.strip()
    m = re.match(r"^(github|gitlab|bitbucket):([\w.-]+)/([\w.-]+)$", u)
    if m:
        return f"{m[1]}.{'org' if m[1] == 'bitbucket' else 'com'}/{m[2]}/{m[3]}".removesuffix(".git").lower()
    if re.match(r"^[\w.-]+/[\w.-]+$", u):
        return f"github.com/{u}".removesuffix(".git").lower()
    m = re.search(r"(github\.com|gitlab\.com|bitbucket\.org)[/:]([\w.-]+)/([\w.-]+?)(\.git)?(/|#|$)", u, re.I)
    if m:
        return f"{m[1]}/{m[2]}/{m[3]}".lower()
    return None


def out_path(repo: str):
    return GIT_LOGS / (repo.replace("/", "__") + ".json.gz")


def extract(repo: str, timeout: int) -> dict:
    tmp = tempfile.mkdtemp(prefix="tr_git_")
    # Never prompt: public repos need no credentials, and a credential-manager pop-up
    # would block the worker until timeout on every private or deleted repository.
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="echo", GCM_INTERACTIVE="never")
    try:
        # GitHub keeps the stored token (anonymous clones get throttled); other hosts get no
        # credentials at all, so private or deleted repos fail immediately.
        no_creds = [] if repo.startswith("github.com/") else ["-c", "credential.helper="]
        clone = subprocess.run(
            ["git", *no_creds, "clone", "--bare", "--filter=blob:none", "--single-branch", "--quiet",
             f"https://{repo}.git", tmp],
            capture_output=True, timeout=timeout, env=env)
        if clone.returncode != 0:
            err = clone.stderr.decode(errors="replace")[-300:]
            return {"repo": repo, "error": "clone", "detail": err}
        log = subprocess.run(
            ["git", "-C", tmp, "log", "--no-merges", "--no-renames",
             "--format=\x1e%at\x1f%ct\x1f%ae\x1f%an\x1f%ce\x1f%cn", "--name-only"],
            capture_output=True, timeout=timeout, env=env)
        if log.returncode != 0:
            return {"repo": repo, "error": "log", "detail": log.stderr.decode(errors="replace")[-300:]}
        commits = []
        for chunk in log.stdout.decode("utf-8", errors="replace").split("\x1e")[1:]:
            head, _, files = chunk.partition("\n")
            at, ct, ae, an, ce, cn = (head.split("\x1f") + [""] * 6)[:6]
            mask, nfiles = 0, 0
            for f in files.splitlines():
                if f.strip():
                    mask |= file_kind(f.strip())
                    nfiles += 1
            bot = bool(BOT.search(ae) or BOT.search(an))
            commits.append([int(at or 0), int(ct or 0), ident(ae, an), ident(ce, cn),
                            mask, min(nfiles, 65535), int(bot)])
        commits.sort()
        return {"repo": repo, "commits": commits}
    except subprocess.TimeoutExpired:
        return {"repo": repo, "error": "timeout"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save(rec):
    p = out_path(rec["repo"])
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    with open(args.repos, encoding="utf-8") as f:
        repos = [l.strip() for l in f if l.strip()]
    todo = [r for r in dict.fromkeys(repos) if not out_path(r).exists()]
    print(f"{len(repos)} repos, {len(todo)} to fetch", flush=True)
    t0, stats = time.time(), {"ok": 0, "err": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(extract, r, args.timeout): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            del futs[fut]          # completed futures would otherwise keep every commit list alive
            save(rec)
            stats["err" if "error" in rec else "ok"] += 1
            if i % 100 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"{i}/{len(todo)} {stats} {rate:.2f}/s eta {(len(todo) - i) / rate / 60:.1f} min",
                      flush=True)


if __name__ == "__main__":
    main()
