"""Filesystem layout shared by every stage of the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODEL_DIR = ROOT / "model"
LOGS = ROOT / "logs"

OSV_ZIP = RAW / "osv" / "npm_all.zip"
UNIVERSE = RAW / "universe.jsonl"
PACKUMENTS = RAW / "packuments"          # one compact .json.gz per package
GIT_LOGS = RAW / "git"                   # one compact .json.gz per repository

for _p in (RAW, INTERIM, PROCESSED, MODEL_DIR, LOGS, PACKUMENTS, GIT_LOGS):
    _p.mkdir(parents=True, exist_ok=True)


def safe_name(pkg: str) -> str:
    """Filesystem-safe file stem for an npm package name (handles scopes)."""
    return pkg.replace("/", "__").replace("@", "_at_")
