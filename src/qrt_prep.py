"""Data prep for the QRT electricity price challenge.

Design rule: anything *fitted* must be fittable inside a CV fold. This module therefore holds
only primitives -- loading, column manifests, fold generation, scoring -- plus fold-safe
transforms that take a training subset and apply it to held-out rows. Nothing here fits on
the full training set and then leaks into validation.

Facts from notebooks/eda.ipynb that this encodes:
  * A day's FR and DE rows are bit-identical across all 32 features; only COUNTRY and TARGET
    differ. Every model must therefore be fitted per country.
  * DAY_ID is shuffled, so it is a grouping key, never a feature.
  * Three columns are exact sign-flips of others and are dropped.
  * The metric is Spearman over FR and DE pooled, so predictions are scored jointly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ID_COLS = ["ID", "DAY_ID", "COUNTRY"]
COUNTRIES = ["FR", "DE"]

# Exact sign-flips: x_NET_IMPORT == -x_NET_EXPORT, FR_DE_EXCHANGE == -DE_FR_EXCHANGE.
# Keeping both makes the design matrix singular (see eda.ipynb 4).
REDUNDANT = ["DE_NET_IMPORT", "FR_NET_IMPORT", "FR_DE_EXCHANGE"]
SIGN_FLIP_PAIRS = [("DE_NET_EXPORT", "DE_NET_IMPORT"),
                   ("FR_NET_EXPORT", "FR_NET_IMPORT"),
                   ("DE_FR_EXCHANGE", "FR_DE_EXCHANGE")]


# --------------------------------------------------------------------------- loading

def load_raw(raw_dir="data/raw"):
    raw = Path(raw_dir)
    return (pd.read_csv(raw / "X_train.csv"),
            pd.read_csv(raw / "y_train.csv"),
            pd.read_csv(raw / "X_test_final.csv"))


def build_frames(raw_dir="data/raw", drop_redundant=True):
    """Raw CSVs -> (train, test). NaNs are preserved; imputation happens inside folds."""
    X, y, Xt = load_raw(raw_dir)
    train = X.merge(y, on="ID").reset_index(drop=True)
    test = Xt.reset_index(drop=True).copy()
    if drop_redundant:
        train = train.drop(columns=REDUNDANT)
        test = test.drop(columns=REDUNDANT)
    return train, test


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS + ["TARGET"]]


# --------------------------------------------------------------------------- validation

def validate(train, test):
    """Re-assert every structural invariant the EDA relied on. Raises on regression."""
    checks = {}
    feats = feature_columns(train)

    checks["no shared days"] = len(set(train.DAY_ID) & set(test.DAY_ID)) == 0
    checks["no shared IDs"] = len(set(train.ID) & set(test.ID)) == 0
    checks["same features"] = feats == feature_columns(test)
    checks["no DE-only days"] = all(
        set(D[D.COUNTRY == "DE"].DAY_ID) <= set(D[D.COUNTRY == "FR"].DAY_ID) for D in (train, test))

    # The crux: a day's two rows differ only in COUNTRY.
    for name, D in [("train", train), ("test", test)]:
        p = D[D.DAY_ID.isin(D.loc[D.COUNTRY == "DE", "DAY_ID"])]
        fr = p[p.COUNTRY == "FR"].sort_values("DAY_ID").set_index("DAY_ID")[feats]
        de = p[p.COUNTRY == "DE"].sort_values("DAY_ID").set_index("DAY_ID")[feats]
        checks[f"{name}: FR/DE rows identical"] = bool((fr - de).abs().max().fillna(0).max() < 1e-12)

    checks["redundant cols dropped"] = not (set(REDUNDANT) & set(train.columns))
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"data prep invariants failed: {failed}")
    return checks


# --------------------------------------------------------------------------- CV + scoring

def make_folds(day_id, seed=0, n_splits=5):
    """Grouped folds keyed on DAY_ID, with the group->fold map randomised by `seed`.

    sklearn's GroupKFold bins groups by size and ignores row order, so it returns the same
    partition every time and gives no error bars. Randomising the assignment fixes that.
    """
    day_id = pd.Series(np.asarray(day_id))
    days = day_id.unique()
    perm = np.random.default_rng(seed).permutation(days)
    fold_of = day_id.map(dict(zip(perm, np.arange(len(perm)) % n_splits))).values
    return [(np.where(fold_of != i)[0], np.where(fold_of == i)[0]) for i in range(n_splits)]


def pooled_spearman(pred, target):
    """The competition metric: one Spearman over FR and DE ranked together."""
    return spearmanr(np.asarray(pred), np.asarray(target)).statistic


def rank_transform(v):
    """Map a target to uniform ranks in (0, 1].

    The largest single win in the EDA (+0.03). 6-8 days carry ~31% of the squared-error
    weight but ~1% of the rank positions, so least squares optimises the wrong thing.
    The metric only sees ranks, so any monotone transform of the target is free.
    """
    v = np.asarray(v, dtype=float)
    return rankdata(v) / len(v)


def impute(fit_df, *apply_dfs, cols=None, strategy="median"):
    """Fill NaNs using statistics computed on `fit_df` only.

    Missingness is day-blocked, matched between train and test, and carries no target signal,
    so the choice barely matters (zero / mean / median all land within noise). Median is used
    because it is fold-safe and robust.
    """
    cols = cols if cols is not None else feature_columns(fit_df)
    fill = 0.0 if strategy == "zero" else getattr(fit_df[cols], strategy)()
    return tuple(D[cols].fillna(fill) for D in (fit_df, *apply_dfs))


# --------------------------------------------------------------------------- selection

def select_by_permutation_null(sub, cols, seed=0, n_perm=2000, alpha=0.05):
    """Features whose |Spearman| with TARGET beats a family-wise permutation null.

    Controls for having scanned every column: the threshold is the (1-alpha) quantile of the
    *maximum* |rho| under a shuffled target. Vectorised over permutations with JAX.

    Call this inside each fold on the training rows only -- selecting on the full training set
    inflates the CV estimate by roughly 0.008 (see data_prep.ipynb 5).
    """
    import jax.numpy as jnp
    from jax import random, vmap

    M = sub[cols].astype("float64")
    Xm = jnp.asarray(M.fillna(M.median()).values)
    yv = jnp.asarray(np.asarray(sub.TARGET.values, dtype=float))

    observed = np.abs(np.asarray(_spearman_columns(Xm, yv)))
    n = Xm.shape[0]
    perm = random.permutation(random.PRNGKey(seed),
                              jnp.broadcast_to(jnp.arange(n), (n_perm, n)),
                              axis=1, independent=True)
    null = np.asarray(vmap(lambda i: _spearman_columns(Xm, yv[i]))(perm))
    threshold = np.quantile(np.abs(null).max(axis=1), 1 - alpha)

    chosen = [c for c, o in zip(cols, observed) if o > threshold]
    return chosen or [cols[int(np.argmax(observed))]]   # never hand back an empty design


def _make_spearman_columns():
    import jax.numpy as jnp
    from jax import jit

    @jit
    def _ranks(x):                      # double argsort; ties arbitrary, fine for continuous data
        order = jnp.argsort(x, axis=0)
        return jnp.zeros_like(order).at[order, jnp.arange(x.shape[1])[None, :]].set(
            jnp.arange(x.shape[0])[:, None]).astype(jnp.float32)

    @jit
    def spearman_columns(Xm, yv):
        rx, ry = _ranks(Xm), _ranks(yv[:, None])[:, 0]
        rx = (rx - rx.mean(0)) / rx.std(0)
        ry = (ry - ry.mean()) / ry.std()
        return (rx * ry[:, None]).mean(0)

    return spearman_columns


_spearman_columns = _make_spearman_columns()


# --------------------------------------------------------------------------- reference pipeline

def cross_validate(train, fit_predict, seeds=range(10), n_splits=5):
    """Pooled out-of-fold Spearman, averaged over randomised fold assignments.

    `fit_predict(train_rows, valid_rows, country) -> predictions`, called once per country per
    fold. Scores by pooling every OOF prediction into a single Spearman, which is what the
    leaderboard does -- averaging per-fold scores would hide cross-country miscalibration.
    """
    scores, oof_last = [], None
    for seed in seeds:
        oof = np.zeros(len(train))
        for tr, va in make_folds(train.DAY_ID, seed=seed, n_splits=n_splits):
            T, V = train.iloc[tr], train.iloc[va]
            for country in COUNTRIES:
                t = T[T.COUNTRY == country]
                mask = (V.COUNTRY == country).values
                if not mask.any():
                    continue
                oof[va[mask]] = fit_predict(t, V[mask], country)
        scores.append(pooled_spearman(oof, train.TARGET))
        oof_last = oof
    return float(np.mean(scores)), float(np.std(scores)), oof_last


def make_submission(test, pred, path):
    """Write an ID/TARGET submission. Order and dtype follow the challenge spec."""
    out = pd.DataFrame({"ID": test.ID.values, "TARGET": np.asarray(pred, dtype=float)})
    if out.TARGET.isna().any():
        raise ValueError("submission contains NaN predictions")
    if len(out) != len(test):
        raise ValueError("submission row count does not match the test set")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- artifacts

def save(train, test, manifest, out_dir="data/processed"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_processed(out_dir="data/processed"):
    out = Path(out_dir)
    return (pd.read_parquet(out / "train.parquet"),
            pd.read_parquet(out / "test.parquet"),
            json.loads((out / "manifest.json").read_text()))
