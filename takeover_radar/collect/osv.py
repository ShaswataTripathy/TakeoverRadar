"""
Parse OSV npm advisories into a table of malicious-version reports.

Sources: OSSF malicious-packages (MAL-*) and GitHub malware advisories
(GHSA with CWE-506 / "malware" in the summary). Most reports are packages
that were malicious from day one (typosquats, dependency confusion); the
compromise filter in build/labels.py separates genuine takeovers of
previously legitimate packages using registry history.

Usage: python -m takeover_radar.collect.osv
"""

import json
import zipfile

from takeover_radar.paths import INTERIM, OSV_ZIP


def _is_malware_ghsa(d):
    cwes = (d.get("database_specific") or {}).get("cwe_ids") or []
    summary = (d.get("summary") or "").lower()
    return "CWE-506" in cwes or "malware" in summary or "malicious code" in summary \
        or "compromised" in summary or "embedded malicious" in summary


def parse():
    z = zipfile.ZipFile(OSV_ZIP)
    out = []
    for n in z.namelist():
        if not (n.startswith("MAL-") or n.startswith("GHSA-")):
            continue
        d = json.loads(z.read(n))
        if d.get("withdrawn"):
            continue
        if n.startswith("GHSA-") and not _is_malware_ghsa(d):
            continue
        for a in d.get("affected") or []:
            pkg = a.get("package") or {}
            if pkg.get("ecosystem") != "npm":
                continue
            introduced, fixed = [], []
            for r in a.get("ranges") or []:
                for e in r.get("events") or []:
                    if "introduced" in e:
                        introduced.append(e["introduced"])
                    if "fixed" in e or "last_affected" in e:
                        fixed.append(e.get("fixed") or e.get("last_affected"))
            out.append({
                "id": d["id"],
                "aliases": d.get("aliases") or [],
                "name": pkg["name"],
                "published": d.get("published"),
                "summary": (d.get("summary") or "")[:300],
                "versions": a.get("versions") or [],
                "introduced": introduced,
                "fixed": fixed,
                "whole_package": not a.get("versions") and introduced == ["0"] and not fixed,
            })
    return out


def main():
    rows = parse()
    path = INTERIM / "osv_reports.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cand = sorted({r["name"] for r in rows if not r["whole_package"]})
    with open(INTERIM / "candidate_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cand) + "\n")
    print(f"{len(rows)} reports, {len(cand)} candidate packages with specific versions")


if __name__ == "__main__":
    main()
