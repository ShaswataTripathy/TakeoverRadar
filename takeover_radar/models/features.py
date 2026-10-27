"""Feature groups and shared preprocessing for every model."""

import numpy as np
import pandas as pd

GROUPS = {
    "registry": "r_",
    "corpus": "c_",
    "git": "g_",
    "contagion": "x_",
    "seq": "s_",          # pretrained sequence features (models/pretrain.py)
}

# Split points (month index, 0 = 2015-01)
DEV_END = (2026 - 2015) * 12          # 2026-01: dev period is everything before it
TEST_START = DEV_END


def feature_cols(df, groups=("registry", "corpus", "git", "contagion")):
    prefixes = tuple(GROUPS[g] for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]


def month_offset(train: pd.DataFrame) -> pd.Series:
    """Empirical log-odds of a takeover per calendar month (the baseline hazard).

    Passed to the models as a fixed offset so they learn *relative* risk: which
    packages are more exposed than others in the same month, not when worms strike.
    """
    g = train.groupby("month")["y"].agg(["sum", "size"])
    p = (g["sum"] + 0.5) / (g["size"] + 1.0)
    return np.log(p / (1 - p))


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))
