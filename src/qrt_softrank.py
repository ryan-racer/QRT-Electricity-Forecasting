"""Differentiable Spearman surrogate in JAX.

Spearman is piecewise-constant in the model scores, so its gradient is zero almost
everywhere and it cannot be optimised directly. This module replaces the hard ranks with
soft ones:

    rank_i  ~=  sum_j sigmoid((s_i - s_j) / tau)

which is smooth, exact in the tau -> 0 limit, and O(n^2) -- fine here, since the largest
per-country training fold is ~680 rows.

The objective is the *pooled* rank correlation across FR and DE, not two separate
per-country objectives. That matches the competition metric exactly, and it makes the
model place both countries' scores on one common scale rather than relying on them
happening to be comparable (the calibration trap in eda.ipynb 6.5).

RESULT: this loses to ridge on a rank-transformed target. Measured over 20 randomised
group-CV seeds, pooled out-of-fold Spearman:

    ridge,         rank target, FR=3 features   0.2904 +/- 0.0063
    ridge,         rank target, FR=29 features  0.2759 +/- 0.0078
    soft-Spearman, lam=0.3,     FR=29 features  0.2714 +/- 0.0146
    soft-Spearman, lam=1.0,     FR=29 features  0.2685 +/- 0.0159
    soft-Spearman, lam=1.0,     FR=3  features  0.2601 +/- 0.0136

(The FR=3 rows use a feature set selected on the full training set, so both are equally
optimistic; the honest fold-internal ridge number is ~0.2806. Either way ridge wins.)

Like-for-like on 29 features the gap is narrow (0.2759 vs 0.2714), and it widens once
France's feature set is cut. The variance gap is the more telling number: ridge is more
than twice as stable across fold seeds.

It converges fine -- the objective plateaus by ~step 1000 at ~0.35 Spearman in-sample --
and it optimises exactly what it is asked to. It simply generalises worse. Two reasons,
both consistent with the rest of this project: rank-transforming the *target* already
captures most of the available benefit, via a closed-form solve with built-in alpha
selection by leave-one-out; and the extra freedom of a non-convex objective fitted by
Adam is capacity, which this dataset reliably punishes.

Kept because the machinery is reusable -- for a different regulariser, a non-linear score
function, or a pairwise/listwise variant -- not because it is the recommended model.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import rankdata


def soft_rank(scores, tau):
    """Differentiable rank. Scores are standardised first, so tau is in units of sd."""
    s = (scores - scores.mean()) / (scores.std() + 1e-8)
    return jax.nn.sigmoid((s[:, None] - s[None, :]) / tau).sum(1)


def neg_soft_spearman(pred, target_rank, tau):
    """Negated Spearman surrogate: Pearson correlation of soft ranks against true ranks."""
    r = soft_rank(pred, tau)
    r = (r - r.mean()) / (r.std() + 1e-8)
    y = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-8)
    return -(r * y).mean()


@partial(jax.jit, static_argnames=("steps", "tau"))
def fit(X_fr, X_de, target_rank, mask_fr, mask_de, lam,
        tau=0.02, steps=1200, lr=0.05, seed=0):
    """Fit per-country linear scores against the pooled soft-Spearman objective.

    Weights need a random init: at zero every prediction is identical, so the score
    standardisation inside `soft_rank` divides 0/0 and the loss goes NaN.
    """
    d = X_fr.shape[1]
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    params = {"wfr": 0.01 * jax.random.normal(k1, (d,)),
              "wde": 0.01 * jax.random.normal(k2, (d,)),
              "bfr": 0.0, "bde": 0.0}

    def loss(p):
        wfr, wde = p["wfr"] * mask_fr, p["wde"] * mask_de
        pred = jnp.concatenate([X_fr @ wfr + p["bfr"], X_de @ wde + p["bde"]])
        return (neg_soft_spearman(pred, target_rank, tau)
                + lam * (jnp.sum(wfr ** 2) + jnp.sum(wde ** 2)))

    opt = optax.adam(lr)

    def step(carry, _):
        p, state = carry
        updates, state = opt.update(jax.grad(loss)(p), state, p)
        return (optax.apply_updates(p, updates), state), None

    (params, _), _ = jax.lax.scan(step, (params, opt.init(params)), None, length=steps)
    return {"wfr": params["wfr"] * mask_fr, "wde": params["wde"] * mask_de,
            "bfr": params["bfr"], "bde": params["bde"]}


def cross_validate(train, features, lam=1.0, tau=0.02, steps=1200,
                   cols_fr=None, cols_de=None, seeds=range(10), n_splits=5):
    """Pooled out-of-fold Spearman for the soft-rank model.

    This does not reuse `qrt_prep.cross_validate`, whose `fit_predict` hook is called once
    per country with rows already filtered to that country. The pooled objective needs
    both countries in the same optimisation, so the fold loop lives here instead.
    """
    import qrt_prep as P

    mask = lambda cols: jnp.asarray([1.0 if (cols is None or c in cols) else 0.0
                                     for c in features])
    m_fr, m_de = mask(cols_fr), mask(cols_de)

    scores, oof_last = [], None
    for seed in seeds:
        oof = np.zeros(len(train))
        for tr, va in P.make_folds(train.DAY_ID, seed=seed, n_splits=n_splits):
            T, V = train.iloc[tr], train.iloc[va]
            t_fr, t_de = T[T.COUNTRY == "FR"], T[T.COUNTRY == "DE"]

            med = T[features].median()
            filled = T[features].fillna(med)
            mu, sd = filled.mean(), filled.std().replace(0, 1)
            z = lambda D: jnp.asarray(((D[features].fillna(med) - mu) / sd).values)

            y = np.concatenate([t_fr.TARGET.values, t_de.TARGET.values])
            ranks = jnp.asarray(rankdata(y) / len(y))
            params = fit(z(t_fr), z(t_de), ranks, m_fr, m_de, lam, tau, steps)

            for country, w, b in [("FR", params["wfr"], params["bfr"]),
                                  ("DE", params["wde"], params["bde"])]:
                m = (V.COUNTRY == country).values
                if m.any():
                    oof[va[m]] = np.asarray(z(V[m]) @ w + b)

        scores.append(P.pooled_spearman(oof, train.TARGET))
        oof_last = oof
    return float(np.mean(scores)), float(np.std(scores)), oof_last
