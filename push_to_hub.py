"""
Push TakeoverRadar to Hugging Face Hub.

Uploads the deployable model (model/), the code score.py needs to run, the
corpus aggregates and labelled events it reads at scoring time, the results
tables and README.md as the model card. Raw packuments and git histories are
not uploaded.

Usage:
    python push_to_hub.py --repo_id tripathyShaswata/TakeoverRadar

Requirements:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent


def push(repo_id: str, private: bool = False):
    api = HfApi()

    print(f"Creating repo {repo_id} (private={private})...")
    create_repo(repo_id, exist_ok=True, private=private)

    print("Uploading model/ ...")
    api.upload_folder(folder_path=ROOT / "model", path_in_repo="model", repo_id=repo_id,
                      repo_type="model", ignore_patterns=["*_dev.txt"])
    print("Uploading code ...")
    api.upload_folder(folder_path=ROOT / "takeover_radar", path_in_repo="takeover_radar",
                      repo_id=repo_id, repo_type="model", ignore_patterns=["__pycache__/*", "*.pyc"])
    print("Uploading results ...")
    api.upload_folder(folder_path=ROOT / "results", path_in_repo="results", repo_id=repo_id,
                      repo_type="model", allow_patterns=["*.csv", "*.json"],
                      ignore_patterns=["*smoke*", "*_v1.csv", "*_v4.csv", "*leaky*"])
    for src, dst in [("score.py", "score.py"), ("README.md", "README.md"),
                     ("data/interim/corpus.pkl", "data/interim/corpus.pkl"),
                     ("data/interim/events.jsonl", "data/interim/events.jsonl"),
                     ("data/interim/mined_events.jsonl", "data/interim/mined_events.jsonl")]:
        print(f"Uploading {src} ...")
        api.upload_file(path_or_fileobj=ROOT / src, path_in_repo=dst, repo_id=repo_id, repo_type="model")

    print(f"\nDone! Model live at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="tripathyShaswata/TakeoverRadar", help="HF repo id")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    push(args.repo_id, args.private)
