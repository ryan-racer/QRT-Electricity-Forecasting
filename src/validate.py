"""Reproduce the cross-validation score behind the submission.

    uv run python src/validate.py             # 5 seeds, about a minute
    uv run python src/validate.py --seeds 16  # the figure quoted in the README, about 3 minutes

For each seed: split the training days into 5 groups at random, fit the full pipeline on
four groups, predict the fifth, and pool every held-out prediction into one Spearman -- the
same quantity the leaderboard computes. The seed only changes which days share a group.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import qrt_prep as P                                                        # noqa: E402
from make_submission import (LGBM, RIDGE, build_features, per_country,     # noqa: E402
                             pooled_catboost, zscore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of random fold assignments")
    args = ap.parse_args()

    train, _, cols, order = build_features()
    n = len(train)
    fr = (train.COUNTRY == "FR").values
    scores = []

    for seed in range(args.seeds):
        ridge, boost, pooled = np.zeros(n), np.zeros(n), np.zeros(n)
        for tr_idx, va_idx in P.make_folds(train.DAY_ID, seed=seed):
            T, V = train.iloc[tr_idx], train.iloc[va_idx]
            ridge[va_idx] = per_country(T, V, cols, order, RIDGE)
            boost[va_idx] = per_country(T, V, cols, order, LGBM)
            pooled[va_idx] = pooled_catboost(T, V, cols)

        # combine exactly as make_submission does, over the full out-of-fold vector
        pred = (zscore(ridge) * 0.7 + zscore(boost) * 0.3) * 0.8 + zscore(pooled) * 0.2
        mu = pred[fr].mean()
        pred[fr] = (pred[fr] - mu) * 0.5 + mu

        score = P.pooled_spearman(pred, train.TARGET)
        scores.append(score)
        print(f"seed {seed:>2}   {score:.4f}", flush=True)

    print(f"\npooled OOF Spearman, {args.seeds} seeds:  {np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
