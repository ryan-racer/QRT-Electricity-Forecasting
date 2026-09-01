"""Recover the true day ordering from the ID column, and build features from it.

THE FINDING. `DAY_ID` is a genuine shuffle -- feature autocorrelation along it is ~0.02.
But `ID` is not. Sorting by `ID` recovers real calendar order:

    feature            by ID    by DAY_ID
    FR_NUCLEAR         0.980      -0.004
    FR_CONSUMPTION     0.965       0.002
    FR_RESIDUAL_LOAD   0.939      -0.013
    FR_SOLAR           0.872      -0.013
    FR_TEMP            0.754       0.015
    GAS_RET            0.180      -0.016

The internal consistency is what makes this certain rather than suggestive: physical
*levels* autocorrelate at 0.87-0.98 while commodity *returns* sit near zero, which is
exactly how real daily data behaves and is not something a shuffle can fake.

THE EXACT MAPPING. `ID` is two stacked blocks over one chronological index:
  FR rows: ID 932..2147  ->  POSITION = ID - 932        (all 1216 days)
  DE rows: ID   0.. 931  ->  POSITION = ID + 284        (a DE row exists iff POSITION >= 284)
so a day's twin rows differ by exactly 1216, and twin features are bit-identical.
Train and test days are interleaved in time, so every test day sits among labelled days.

`ID % 1216` also orders the days, but only up to a CYCLIC ROTATION: it sends the first 284
days to the end, creating one false adjacency at the join. 1215 of 1216 adjacencies survive,
so it still scores well, but rolling windows straddling the seam are wrong -- under it the
largest 1-step jump in FR_NUCLEAR is 6.39 against 3.82 under the correct order (median 0.08).

POSITION 1215 (DAY_ID 190, a test row) is a partial/corrupt final day: it sits at the global
minimum of several unrelated columns at once and its total 1-step change is 59.3 against a
median of 13.1. Its difference features are an artefact; `clip_outliers` handles it.

PROVENANCE. This is the "leak" the challenge winner describes in the Collège de France
seminar of 31 Jan 2024 (notes/winner_talk.md). He reached 0.32 without it via cluster-based
de-seasonalising, then used first differences and deviations from trailing moving averages
of the ID-ordered series to make the jump. QRT's own representative acknowledges it on the
recording: the data was anonymised such that the test set could still be sorted.

NO TARGET IS INVOLVED. Every feature here is built from X alone, and X is supplied for the
test set too, so these features are computable at submission time exactly as they are in
cross-validation. Rolling statistics use `.shift(1)` so a day never sees its own value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_DAYS = 1216


def day_index(df):
    """True chronological day POSITION (0..1215) recovered from ID.

    FR rows carry ID 932..2147, DE rows ID 0..931, over one shared chronological index.
    """
    return np.where(df.COUNTRY.values == "FR", df.ID.values - 932, df.ID.values + 284)


def add_time_features(train, test, feature_cols, lags=(1, 2), windows=(7, 30),
                      include_lagged_levels=False, clip_outliers=8.0):
    """Return (train, test) with ID-ordering features appended.

    For each country's series, ordered by the recovered day index:
      <col>_d<L>   first difference at lag L        (change since L days ago)
      <col>_ma<W>  value minus trailing mean over W past days   (deviation from regime)
      <col>_l<L>   the lagged level itself          (optional; adds many columns)

    Args:
        train, test: frames from qrt_prep.build_frames()
        feature_cols: base columns to derive from
        lags, windows: horizons, in days
        include_lagged_levels: also emit raw lagged levels

    The two frames are concatenated before differencing so that a test day's history can
    include train days and vice versa -- they are interleaved in time, and using test *X*
    is legitimate because the organisers supply it.
    """
    both = pd.concat([train.drop(columns=[c for c in ["TARGET"] if c in train]), test],
                     ignore_index=True)
    both["_t"] = day_index(both)

    pieces = []
    for country in ("FR", "DE"):
        s = both[both.COUNTRY == country].sort_values("_t").copy()
        for col in feature_cols:
            v = s[col]
            for L in lags:
                s[f"{col}_d{L}"] = v.diff(L)
                if include_lagged_levels:
                    s[f"{col}_l{L}"] = v.shift(L)
            for W in windows:
                # shift(1) first: a day must never enter its own trailing window
                s[f"{col}_ma{W}"] = v - v.shift(1).rolling(W, min_periods=2).mean()
        pieces.append(s)

    derived = pd.concat(pieces, ignore_index=True)
    new_cols = [c for c in derived.columns if c not in both.columns]

    if clip_outliers:
        # POSITION 1215 is a corrupt final day whose differences are a ~60-sigma artefact.
        # Clip in units of each column's own sd rather than dropping the row: it is a test
        # day, so it must still receive a prediction.
        d = derived[new_cols]
        lim = clip_outliers * d.std()
        derived[new_cols] = d.clip(-lim, lim, axis=1)

    keep = ["ID"] + new_cols
    return (train.merge(derived[keep], on="ID", how="left"),
            test.merge(derived[keep], on="ID", how="left"),
            new_cols)


def audit_ordering(train, test, feature_cols):
    """Evidence the ID ordering is real. Uses features only -- never the target."""
    both = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
    both["_t"] = day_index(both)
    fr = both[both.COUNTRY == "FR"]
    rows = []
    for c in feature_cols:
        rows.append({
            "feature": c,
            "by_ID": fr.sort_values("_t")[c].reset_index(drop=True).autocorr(1),
            "by_DAY_ID": fr.sort_values("DAY_ID")[c].reset_index(drop=True).autocorr(1),
            "shuffled": fr[c].sample(frac=1, random_state=0).reset_index(drop=True).autocorr(1),
        })
    return pd.DataFrame(rows).sort_values("by_ID", ascending=False)
