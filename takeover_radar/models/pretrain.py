"""
Self-supervised pretraining over package histories, used as features for the GBM.

Labels are scarce (a few hundred takeovers) but unlabelled history is not
(millions of package-months). A causal transformer learns to predict each
package's next monthly feature vector from its past. Two things come out of it
for every package-month t, both computed only from tokens 1..t, i.e. from
information published before month t began:

  s_surprise*  how badly the history up to t-1 predicted month t's features
               (overall and per feature family). A takeover-in-progress looks
               like a break from the package's own rhythm: new hands, new
               cadence, new publishing path.
  s_emb_*      a PCA-compressed summary of the history up to t.

Pretraining sees only months before the test period, so it never touches the
held-out year, and it never sees labels.

Usage: python -m takeover_radar.models.pretrain     (writes data/processed/seq_feats.parquet)
       python -m takeover_radar.models.pretrain --reuse   (re-extract with the saved encoder)
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from takeover_radar.models.features import DEV_END, GROUPS, feature_cols
from takeover_radar.models.seq import DEVICE, HazardTransformer, SeqData
from takeover_radar.paths import MODEL_DIR, PROCESSED

N_EMB = 8


class NextStep(nn.Module):
    def __init__(self, dim_in, n_feat, d=128, layers=4, dropout=0.1):
        super().__init__()
        self.body = HazardTransformer(dim_in, d=d, layers=layers, dropout=dropout)
        self.next = nn.Linear(d, n_feat)

    def encode(self, tok, pos, valid):
        b = self.body
        h = b.inp(tok) + b.age(pos)
        L = tok.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tok.device), 1)
        return b.norm(b.enc(h, mask=causal, src_key_padding_mask=~valid))


def _targets(tok, n_feat):
    """Next-month standardized features and which of them were observed."""
    x = tok[:, 1:, :n_feat]
    obs = 1.0 - tok[:, 1:, 2 * n_feat:3 * n_feat]
    return x, obs


def pretrain(data, n_feat, epochs=4, batch_size=192, lr=5e-4, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NextStep(data.dim, n_feat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    idx_all = np.where(data.ends - data.starts >= 2)[0]
    steps = len(idx_all) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps, pct_start=0.05)
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(idx_all)
        t0, tot = time.time(), 0.0
        for k in range(steps):
            tok, _, _, pos, valid, _ = data.batch(idx[k * batch_size:(k + 1) * batch_size])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                h = model.encode(tok, pos, valid)
                pred = model.next(h[:, :-1]).float()
            x, obs = _targets(tok, n_feat)
            m = obs * valid[:, 1:].unsqueeze(-1)
            loss = (((pred - x) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if k % 200 == 0:
                log(f"    ep {ep} step {k}/{steps} mse {tot / (k + 1):.4f} "
                    f"{(k + 1) / (time.time() - t0):.1f} it/s")
        log(f"  epoch {ep}: mse {tot / steps:.4f} ({time.time() - t0:.0f}s)")
    return model


def extract(model, data, n_feat, group_idx, batch_size=256):
    """Per-token surprise (overall + per family) and hidden state, aligned to data's flat order."""
    model.eval()
    n = len(data.y)
    surprise = np.full((n, 1 + len(group_idx)), np.nan, np.float32)
    hidden = np.zeros((n, model.next.in_features), np.float16)
    idx = np.arange(len(data.names))
    idx = idx[np.argsort(-(data.ends - data.starts))]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        for k in range(0, len(idx), batch_size):
            tok, _, _, pos, valid, flat = data.batch(idx[k:k + batch_size], max_len=10_000)
            h = model.encode(tok, pos, valid)
            pred = model.next(h[:, :-1]).float()
            x, obs = _targets(tok, n_feat)
            se = ((pred - x) ** 2) * obs
            cols = [se.sum(-1) / obs.sum(-1).clamp(min=1)]
            for gi in group_idx:
                o = obs[..., gi]
                cols.append(se[..., gi].sum(-1) / o.sum(-1).clamp(min=1))
            s = torch.stack(cols, -1).cpu().numpy()
            hv = h.float().cpu().numpy()
            m = flat >= 0
            hidden[flat[m]] = hv[m]
            # surprise of token t is the error of the prediction made at t-1
            m1 = m[:, 1:]
            surprise[flat[:, 1:][m1]] = s[m1]
    return surprise, hidden


def load_encoder():
    """Saved encoder plus normalisation and PCA, for scoring packages outside the panel."""
    meta = np.load(MODEL_DIR / "pretrain_meta.npz")
    feats = list(meta["feats"])
    model = NextStep(3 * len(feats), len(feats)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "pretrain.pt", map_location=DEVICE))
    return model, meta


def seq_features(df, model, meta):
    """s_ features for every row of df (columns name, month, y?, plus the panel features)."""
    feats = list(meta["feats"])
    gnames = list(meta["groups"])
    group_idx = [[i for i, c in enumerate(feats) if c.startswith(GROUPS[g])] for g in gnames]
    df = df.assign(y=df["y"] if "y" in df else 0).reindex(columns=["name", "month", "y"] + feats)
    data = SeqData(df, feats, (meta["mu"], meta["sd"]))
    surprise, hidden = extract(model, data, len(feats), group_idx)
    emb = (hidden.astype(np.float32) - meta["pca_mu"]) @ meta["pca_vt"].T
    out = pd.DataFrame({"name": np.repeat(data.names, data.ends - data.starts), "month": data.month})
    out["s_surprise"] = surprise[:, 0]
    for j, g in enumerate(gnames, 1):
        out[f"s_surprise_{g}"] = surprise[:, j]
    for j in range(emb.shape[1]):
        out[f"s_emb_{j}"] = emb[:, j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="load pretrain.pt instead of training")
    args = ap.parse_args()
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel = panel[panel["in_universe"] == 1].reset_index(drop=True)
    feats = feature_cols(panel)
    group_idx = [[i for i, c in enumerate(feats) if c.startswith(p)] for p in GROUPS.values()]
    group_idx = [g for g in group_idx if g]
    gnames = [g for g, p in GROUPS.items() if any(c.startswith(p) for c in feats)]

    pre = panel[panel["month"] < DEV_END]
    data = SeqData(pre, feats)
    stats = data.stats
    del pre
    if args.reuse:
        model = NextStep(data.dim, len(feats)).to(DEVICE)
        model.load_state_dict(torch.load(PROCESSED / "pretrain.pt", map_location=DEVICE))
        log("reusing saved encoder")
    else:
        log(f"pretraining on {len(data.y)} rows / {len(data.names)} packages, {len(feats)} features")
        model = pretrain(data, len(feats), epochs=args.epochs, log=log)
    del data

    # Extract over every month (causal, so a row's features only use its own past).
    full = SeqData(panel, feats, stats)
    surprise, hidden = extract(model, full, len(feats), group_idx)
    dev_rows = full.month < DEV_END
    # PCA fitted on pre-test rows only
    rng = np.random.default_rng(0)
    fit_rows = np.where(dev_rows)[0]
    fit_rows = rng.choice(fit_rows, min(500_000, len(fit_rows)), replace=False)
    H = hidden[fit_rows].astype(np.float32)
    mu = H.mean(0)
    _, _, vt = np.linalg.svd(H - mu, full_matrices=False)
    emb = (hidden.astype(np.float32) - mu) @ vt[:N_EMB].T

    out = pd.DataFrame({"name": np.repeat(full.names, full.ends - full.starts), "month": full.month})
    out["s_surprise"] = surprise[:, 0]
    for j, g in enumerate(gnames, 1):
        out[f"s_surprise_{g}"] = surprise[:, j]
    for j in range(N_EMB):
        out[f"s_emb_{j}"] = emb[:, j].astype(np.float32)
    out.to_parquet(PROCESSED / "seq_feats.parquet", index=False)
    if not args.reuse:
        torch.save(model.state_dict(), PROCESSED / "pretrain.pt")
    torch.save(model.state_dict(), MODEL_DIR / "pretrain.pt")
    # everything needed to compute s_ features for a single live package (score.py)
    np.savez(MODEL_DIR / "pretrain_meta.npz", mu=stats[0], sd=stats[1], pca_mu=mu,
             pca_vt=vt[:N_EMB], feats=np.array(feats), groups=np.array(gnames))
    log(f"seq features: {len(out)} rows, {out.shape[1] - 2} columns")


if __name__ == "__main__":
    main()
