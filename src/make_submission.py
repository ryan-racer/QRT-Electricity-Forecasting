"""Reproduce the final submission end to end.

    uv run python src/make_submission.py

Writes submissions/blend3_final.csv. Public leaderboard: 0.5658, rank 6 / 1157.
Local CV (pooled out-of-fold Spearman, 16 randomised day-grouped seeds): 0.5716 +/- 0.0050.

Pipeline:
  1. recover the true day ordering from the ID column          (qrt_timeorder.day_index)
  2. first differences + 7-day trailing deviation of all 29 base features
  3. trailing 5- and 20-day cumulative sums of the commodity returns (France only)
  4. local level / volatility of the 3 nearest training days' targets in time
  5. per-country RidgeCV and LightGBM on the rank-transformed target
  6. one pooled CatBoost with a country indicator, for diversity
  7. z-score, blend 0.56 / 0.24 / 0.20, shrink France's spread by 0.5
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import lightgbm as lgb                                   # noqa: E402
from catboost import CatBoostRegressor                   # noqa: E402
from sklearn.linear_model import RidgeCV                 # noqa: E402

import qrt_prep as P                                     # noqa: E402
import qrt_temporal as TT                                # noqa: E402
import qrt_timeorder as TO                               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submissions" / "blend3_final.csv"

# France gets only the three features that survive a permutation null (eda.ipynb 5.1);
# Germany gets all 29. Giving France everything overfits badly.
FR_BASE = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"]

RIDGE = lambda: RidgeCV(alphas=np.logspace(-2, 4, 40))
LGBM = lambda: lgb.LGBMRegressor(num_leaves=4, n_estimators=400, learning_rate=0.05,
                                 min_child_samples=20, reg_lambda=1.0, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.7,
                                 random_state=0, verbose=-1)
CAT = lambda: CatBoostRegressor(depth=6, iterations=400, learning_rate=0.05, l2_leaf_reg=6,
                                verbose=0, random_seed=0, allow_writing_files=False)

zscore = lambda v: (v - v.mean()) / v.std()


def build_features():
    train, test = P.build_frames(ROOT / "data" / "raw")
    base = P.feature_columns(train)

    train, test, time_cols = TO.add_time_features(train, test, base, lags=(1,), windows=(7,))
    train, test, cum_cols = TO.add_cumulative_returns(train, test, windows=(5, 20))

    for d in (train, test):
        d["_pos"] = TO.day_index(d)
    order = {**dict(zip(train.DAY_ID, train._pos)), **dict(zip(test.DAY_ID, test._pos))}

    cols = {"FR": FR_BASE + time_cols + cum_cols,
            "DE": base + time_cols}
    return train, test, cols, order


def with_neighbours(frame, cols, source, order):
    """Append temporal-neighbour target features. `source` supplies the targets."""
    nb = TT.neighbour_features(frame.DAY_ID.values, source.DAY_ID.values,
                               source.TARGET.values, order, k=3)
    return pd.concat([frame[cols].reset_index(drop=True), nb], axis=1)


def per_country(train, test, cols, order, make_model):
    pred = np.zeros(len(test))
    for country in ("FR", "DE"):
        t = train[train.COUNTRY == country]
        mask = (test.COUNTRY == country).values
        A = with_neighbours(t, cols[country], t, order)
        B = with_neighbours(test[mask], cols[country], t, order)
        med = A.median()
        model = make_model().fit(A.fillna(med), P.rank_transform(t.TARGET.values))
        pred[mask] = model.predict(B.fillna(med))
    return pred


def pooled_catboost(train, test, cols):
    every = sorted(set(cols["FR"]) | set(cols["DE"]))
    A = train[every].copy()
    B = test[every].copy()
    A["is_fr"] = (train.COUNTRY == "FR").astype(int).values
    B["is_fr"] = (test.COUNTRY == "FR").astype(int).values
    med = A.median()
    model = CAT().fit(A.fillna(med), P.rank_transform(train.TARGET.values))
    return model.predict(B.fillna(med))


def main():
    train, test, cols, order = build_features()

    ridge = per_country(train, test, cols, order, RIDGE)
    boost = per_country(train, test, cols, order, LGBM)
    pooled = pooled_catboost(train, test, cols)

    pred = (zscore(ridge) * 0.7 + zscore(boost) * 0.3) * 0.8 + zscore(pooled) * 0.2

    # Germany is far better predicted than France (within-country 0.77 vs 0.28), and the
    # metric ranks both countries in one pool. Compressing France toward its own mean lets
    # the more reliable German predictions occupy the extremes of the pooled ranking.
    fr = (test.COUNTRY == "FR").values
    mu = pred[fr].mean()
    pred[fr] = (pred[fr] - mu) * 0.5 + mu

    sub = P.make_submission(test, pred, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
