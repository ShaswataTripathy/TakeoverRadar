"""
Neural discrete-time hazard model over each package's monthly history.

Each package is a sequence of monthly tokens. Token t holds the features as of
the first day of month t (they already exclude anything published in month t),
their month-over-month change, and a missingness mask. A causal transformer
reads the history up to t and outputs the log-odds of a takeover during month t:

    logit_t = f(token_1..t) + b[calendar month]

The per-month bias absorbs the calendar baseline hazard (worm waves), so f
learns relative risk. Only f is used for ranking.
"""

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from takeover_radar.eval.protocol import event_percentiles
from takeover_radar.models.features import month_offset, signed_log

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SeqData:
    """Packages stored as one flat token array plus offsets; batches are padded on the fly."""

    def __init__(self, df, feats, stats=None):
        df = df.sort_values(["name", "month"], kind="stable")
        X = signed_log(df[feats].to_numpy(np.float32))
        miss = np.isnan(X)
        if stats is None:
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0) + 1e-3
            stats = (np.nan_to_num(mu), np.nan_to_num(sd, nan=1.0))
        X = (X - stats[0]) / stats[1]
        X = np.clip(np.nan_to_num(X, nan=0.0), -8, 8)
        self.stats = stats
        names = df["name"].to_numpy()
        starts = np.r_[0, np.where(names[1:] != names[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(df)]
        dX = np.zeros_like(X)
        dX[1:] = X[1:] - X[:-1]
        dX[starts] = 0.0
        self.tok = np.concatenate([X, dX, miss.astype(np.float32)], axis=1).astype(np.float16)
        self.y = df["y"].to_numpy(np.float32)
        self.month = df["month"].to_numpy(np.int64)
        self.names = names[starts]
        self.starts, self.ends = starts, ends
        self.has_event = np.array([self.y[s:e].max() > 0 for s, e in zip(starts, ends)])
        self.order = df.index.to_numpy()       # original row index per flat token
        self.dim = self.tok.shape[1]

    def batch(self, idx, max_len=128):
        lens = np.minimum(self.ends[idx] - self.starts[idx], max_len)
        L = int(lens.max())
        B = len(idx)
        tok = np.zeros((B, L, self.dim), np.float16)
        y = np.zeros((B, L), np.float32)
        mon = np.zeros((B, L), np.int64)
        pos = np.zeros((B, L), np.int64)
        valid = np.zeros((B, L), bool)
        flat = np.full((B, L), -1, np.int64)
        for b, i in enumerate(idx):
            e = self.ends[i]
            s = max(self.starts[i], e - max_len)   # keep the most recent max_len months
            n = e - s
            tok[b, :n] = self.tok[s:e]
            y[b, :n] = self.y[s:e]
            mon[b, :n] = self.month[s:e]
            pos[b, :n] = np.arange(s - self.starts[i], e - self.starts[i])
            valid[b, :n] = True
            flat[b, :n] = np.arange(s, e)
        t = lambda a: torch.from_numpy(a).to(DEVICE, non_blocking=True)
        return t(tok).float(), t(y), t(mon), t(np.minimum(pos, 255)), t(valid), flat


class HazardTransformer(nn.Module):
    def __init__(self, dim_in, d=128, layers=4, heads=4, dropout=0.1, n_months=160):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(dim_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.age = nn.Embedding(256, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.month_bias = nn.Embedding(n_months, 1)

    def forward(self, tok, pos, valid):
        h = self.inp(tok) + self.age(pos)
        L = tok.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tok.device), 1)
        h = self.enc(h, mask=causal, src_key_padding_mask=~valid)
        return self.head(self.norm(h)).squeeze(-1)


def _pos_weights(data, events):
    """sqrt campaign down-weighting for positive tokens, as in the GBM."""
    camp = events.set_index("name")["campaign"]
    pos_names = data.names[data.has_event]
    sizes = pd.Series(pos_names).map(camp).value_counts()
    w = {n: 1.0 / math.sqrt(sizes[camp[n]]) for n in pos_names if n in camp}
    mean = np.mean(list(w.values())) if w else 1.0
    return {n: v / mean for n, v in w.items()}


def predict(model, data, batch_size=512):
    model.eval()
    out = np.zeros(len(data.y), np.float32)
    idx = np.arange(len(data.names))
    # longest first keeps padding low
    idx = idx[np.argsort(-(data.ends - data.starts))]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        for k in range(0, len(idx), batch_size):
            tok, y, mon, pos, valid, flat = data.batch(idx[k:k + batch_size], max_len=10_000)
            f = model(tok, pos, valid).float().cpu().numpy()
            m = flat >= 0
            out[flat[m]] = f[m]
    return out


def _quick_auc(model, data):
    s = predict(model, data)
    df = pd.DataFrame({"name": np.repeat(data.names, data.ends - data.starts),
                       "month": data.month, "y": data.y, "score": s})
    ev = event_percentiles(df)
    return ev["pct"].mean() if len(ev) else float("nan"), (ev["rank_frac"] <= 0.01).mean()


def fit(train_df, feats, events, val_df=None, epochs=12, batch_size=96, lr=3e-4, pos_boost=8,
        d=128, layers=4, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = SeqData(train_df, feats)
    val = SeqData(val_df, feats, data.stats) if val_df is not None else None
    model = HazardTransformer(data.dim, d=d, layers=layers).to(DEVICE)
    off = month_offset(train_df)
    with torch.no_grad():
        model.month_bias.weight.fill_(float(off.min()))
        model.month_bias.weight[torch.tensor(off.index.to_numpy(), device=DEVICE)] = \
            torch.tensor(off.to_numpy(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    pw = _pos_weights(data, events)
    pkg_w = np.array([pw.get(n, 1.0) for n in data.names], np.float32)

    pos_idx = np.where(data.has_event)[0]
    neg_idx = np.where(~data.has_event)[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    steps_per_epoch = (len(neg_idx) + len(pos_idx) * pos_boost) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        idx = np.concatenate([neg_idx, np.repeat(pos_idx, pos_boost)])
        rng.shuffle(idx)
        t0, tot = time.time(), 0.0
        for k in range(steps_per_epoch):
            b = idx[k * batch_size:(k + 1) * batch_size]
            tok, y, mon, pos, valid, _ = data.batch(b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                f = model(tok, pos, valid).float()
            logit = f + model.month_bias(mon).squeeze(-1)
            w = torch.from_numpy(pkg_w[b]).to(DEVICE).unsqueeze(1).expand_as(y)
            # Positive packages are oversampled, so scores are for ranking, not calibrated.
            w = torch.where(y > 0, w, torch.ones_like(w)) * valid
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / valid.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if k % 200 == 0:
                log(f"    ep {ep} step {k}/{steps_per_epoch} loss {tot / (k + 1):.5f} "
                    f"{(k + 1) / (time.time() - t0):.1f} it/s")
        msg = f"  epoch {ep}: loss {tot / steps_per_epoch:.5f} ({time.time() - t0:.0f}s)"
        if val is not None:
            auc, r1 = _quick_auc(model, val)
            msg += f" val auc {auc:.4f} r@1% {r1:.3f}"
            if auc > best:
                best = auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, data.stats


def score(model, stats, df, feats):
    data = SeqData(df, feats, stats)
    s = predict(model, data)
    out = pd.Series(np.nan, index=df.index)
    out.loc[data.order] = s
    return out.to_numpy()
