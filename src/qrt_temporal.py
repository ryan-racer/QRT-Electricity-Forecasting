"""Turn a recovered day ordering into features, without leaking.

The competition winner reportedly reverse-engineered the temporal ordering of the
anonymised days. If that is right, the payoff is not the ordering itself but what it
unlocks: electricity futures returns show volatility clustering and short-horizon
momentum, so a test day's *temporal neighbours* carry information about it. Train and
test days are interleaved in time, so almost every test day has labelled days near it.

One trap worth naming: measuring TARGET autocorrelation along DAY_ID order returns ~0,
but DAY_ID is a shuffle, so that measures nothing at all. Autocorrelation in *true* time
is an open question, and only a recovered ordering can answer it. (eda.ipynb 1.3 only ever
measures FEATURE autocorrelation along DAY_ID, which is a valid shuffle test and stands.)

THE LEAKAGE RULE, which is the whole game here:
a row's neighbour features may only ever be built from days whose targets are legitimately
known. Inside cross-validation that means the training fold only -- never the validation
fold, and never the row itself. `neighbour_features` enforces this by taking an explicit
set of allowed source days. Get this wrong and the score will look spectacular and be worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_order(path):
    """Read a DAY_ID -> position mapping. Returns a dict."""
    o = pd.read_csv(path)
    col = [c for c in o.columns if c != "DAY_ID"][0]
    return dict(zip(o.DAY_ID.values, o[col].values))


def neighbour_features(target_days, source_days, source_targets, order,
                       k=5, max_gap=None):
    """Build temporal-neighbour features for `target_days`.

    Args:
        target_days:    day ids needing features (validation or test days)
        source_days:    day ids whose targets may be used (TRAINING fold only)
        source_targets: target value for each source day, same order
        order:          dict day_id -> position in recovered time
        k:              number of nearest days in time to aggregate
        max_gap:        if set, ignore source days further than this many positions away

    Returns a DataFrame indexed like `target_days` with neighbour statistics. Rows with
    no usable neighbour get NaN, which downstream imputation handles.
    """
    src = np.array([order.get(d, np.nan) for d in source_days], dtype=float)
    tgt = np.array([order.get(d, np.nan) for d in target_days], dtype=float)
    sy = np.asarray(source_targets, dtype=float)
    sid = np.asarray(source_days)
    tid = np.asarray(target_days)

    ok = ~np.isnan(src)
    src, sy, sid = src[ok], sy[ok], sid[ok]
    sort = np.argsort(src)
    src, sy, sid = src[sort], sy[sort], sid[sort]

    out = np.full((len(tgt), 6), np.nan)
    for i, p in enumerate(tgt):
        if np.isnan(p):
            continue
        # A day must never be its own neighbour. Callers may legitimately pass
        # overlapping source/target sets (e.g. building features for training rows),
        # so drop the identity match here rather than trusting the caller.
        keep = sid != tid[i]
        src_i, sy_i = src[keep], sy[keep]
        gap = np.abs(src_i - p)
        if len(gap) == 0:
            continue
        near = np.argsort(gap)[:k]
        g = gap[near]
        if max_gap is not None:
            near = near[g <= max_gap]
            g = g[g <= max_gap]
        if len(near) == 0:
            continue
        v = sy_i[near]
        w = 1.0 / (1.0 + g)                      # closer days count for more
        before = sy_i[(src_i < p)][-k:]
        after = sy_i[(src_i > p)][:k]
        out[i] = [
            np.average(v, weights=w),            # local level  (momentum)
            np.average(np.abs(v), weights=w),    # local volatility (clustering)
            v.std() if len(v) > 1 else np.nan,
            g.min(),                             # how well-covered this day is
            before.mean() if len(before) else np.nan,
            after.mean() if len(after) else np.nan,
        ]
    return pd.DataFrame(out, columns=["nb_mean", "nb_absmean", "nb_std",
                                      "nb_gap", "nb_before", "nb_after"])


def validate_order(day_frame, order, feature_cols, n_perm=200, seed=0):
    """Evidence that an ordering is real, using FEATURES only -- never the target.

    Real daily series are autocorrelated; a shuffle is not. So sort the days by the
    recovered position and measure lag-1 autocorrelation of each feature. Compare against
    random orderings to get a null. Any column used to BUILD the ordering is circular
    evidence -- pass only held-out columns in `feature_cols`.
    """
    d = day_frame.copy()
    d["_pos"] = d.DAY_ID.map(order)
    d = d.dropna(subset=["_pos"]).sort_values("_pos")

    rng = np.random.default_rng(seed)
    rows = []
    for c in feature_cols:
        v = d[c].astype(float)
        obs = v.autocorr(1)
        null = [v.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).autocorr(1)
                for _ in range(n_perm)]
        null = np.array(null)
        rows.append({"feature": c, "lag1_autocorr": obs,
                     "null_mean": null.mean(), "null_sd": null.std(),
                     "z": (obs - null.mean()) / (null.std() + 1e-12)})
    return pd.DataFrame(rows).sort_values("z", ascending=False)
