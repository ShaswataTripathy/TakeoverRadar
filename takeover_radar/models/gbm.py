"""Gradient-boosted discrete-time hazard model (LightGBM) and simple baselines."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from takeover_radar.models.features import month_offset

PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=200,
    min_sum_hessian_in_leaf=1e-3,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    max_bin=127,
    verbose=-1,
    num_threads=20,
)


def campaign_weights(train, events, mode):
    """Down-weight giant worm waves so one campaign does not define 'risky'."""
    w = np.ones(len(train))
    if mode == "none":
        return w
    size = train.loc[train["y"] == 1, "name"].map(events.set_index("name")["campaign"]).value_counts()
    camp = train["name"].map(events.set_index("name")["campaign"])
    pos = (train["y"] == 1).to_numpy()
    s = camp[pos].map(size).to_numpy(dtype=float)
    w[pos] = 1.0 / np.sqrt(s) if mode == "sqrt" else 1.0 / s
    w[pos] *= pos.sum() / w[pos].sum()   # keep total positive mass unchanged
    return w


def subsample(train, rate, seed):
    rng = np.random.default_rng(seed)
    keep = (train["y"].to_numpy() == 1) | (rng.random(len(train)) < rate)
    out = train[keep]
    w = np.where(out["y"].to_numpy() == 1, 1.0, 1.0 / rate)
    return out, w


def fit(train, feats, events, n_rounds=600, neg_rate=0.25, weight_mode="sqrt", seed=0,
        params=None):
    tr, w_s = subsample(train, neg_rate, seed)
    w = w_s * campaign_weights(tr, events, weight_mode)
    off = tr["month"].map(month_offset(train)).to_numpy()
    ds = lgb.Dataset(tr[feats].to_numpy(np.float32), tr["y"].to_numpy(), weight=w,
                     init_score=off, feature_name=feats, free_raw_data=True)
    p = dict(PARAMS, seed=seed, **(params or {}))
    return lgb.train(p, ds, num_boost_round=n_rounds)


def predict(model, df, feats):
    return model.predict(df[feats].to_numpy(np.float32), raw_score=True)


# ----------------------------------------------------------------------------- baselines

def weak_links_score(df):
    """Hand-built 'weak links' heuristic (single maintainer, custom email domain,
    install scripts, stale, no OIDC/provenance), tie-broken by dependents."""
    s = ((df["r_n_maint"] <= 1).astype(float)
         + df["r_any_custom_domain"].fillna(0)
         + df["r_hooks"].fillna(0)
         + (df["r_days_since_pub"] > 365).astype(float)
         + (1 - df["r_ever_tp"].fillna(0)) * (1 - df["r_ever_att"].fillna(0)))
    return s + 0.01 * df["c_indegree"].fillna(0)


def popularity_score(df):
    return df["c_indegree"].fillna(0) + 1e-3 * df["r_log_n_versions"].fillna(0)


def logistic(train, test, feats, events):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from takeover_radar.models.features import signed_log
    tr, w = subsample(train, 0.1, 0)
    w = w * campaign_weights(tr, events, "sqrt")
    X = signed_log(tr[feats].to_numpy(np.float64))
    clf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000))
    clf.fit(X, tr["y"].to_numpy(), logisticregression__sample_weight=w)
    return clf.decision_function(signed_log(test[feats].to_numpy(np.float64)))
