# Ensembling, stacking and mixture-of-experts

**Regime note.** This study was first run against the old thin-signal baseline (per-country
ridge on the 29 raw features, 0.2909; best blend 0.3005). It was then re-run from scratch on
top of the ID-recovered calendar features (`src/qrt_timeorder.py`), where plain per-country
ridge scores 0.5231. Both sets of results are reported, because the comparison between them is
itself the most useful finding: **almost every conclusion about model choice flipped, and every
conclusion about the pooled metric's geometry survived and hardened.**

| reference point | pooled OOF Spearman (15 seeds) |
|---|---|
| BASELINE A - ridge, 29 raw features, rank target | 0.2909 +/- 0.0069 |
| BASELINE B - 0.5 z(ridge) + 0.5 z(TabPFN v3), raw features | 0.3005 +/- 0.0067 |
| **BASELINE T - ridge + time features (`lags=(1,)`, `windows=(7,)`, 58 new cols)** | **0.5231 +/- 0.0061** |

## Evaluation protocol

All comparisons are **paired on identical fold assignments** (`P.make_folds(seed=0..14)`,
grouped on `DAY_ID`). `d` is the mean per-seed delta against the stated baseline, `sd` its
standard deviation, `w` wins out of 15, `t` the paired t-statistic. The marginal noise band is
+/-0.007; paired noise is far smaller, so a delta of +0.002 at 15/15 is real but negligible,
while -0.010 at 0/15 is a real loss.

**Nesting.** Every level-2 model, gate and blend weight is fitted *inside* the outer fold on
level-1 predictions that are themselves out-of-fold with respect to the level-2 training rows.
The affordable construction: for a 5-fold partition, the honest predictions outer fold `k`
needs on its 4 training folds come from models trained on the complement of `{j,k}`; that set
is symmetric in `j` and `k`, so only `C(5,2)=10` extra level-1 fits per country per seed are
required, each serving two outer folds.

Leakage check: no stacked or gated configuration ever scored above the best single model by
more than its oracle bound, and nothing jumped implausibly. A stacked score above the oracle
per-row pick would have meant the level-2 fit was seeing its own validation rows.

## 1. Structural findings, re-tested in the time-ordered regime

### 1a. Pooled, never per-country, normalisation - SURVIVES and hardens

The metric ranks FR and DE together, so the *relative spread* of the two countries'
predictions is part of the score. Renormalising each country separately destroys it.

| operation on the ridge OOF | old regime (base 0.2909) | time-ordered regime (base 0.5231) |
|---|---|---|
| per-country z-score | -0.0094 | **-0.0484** (0/15, t = -118) |
| per-country uniform rank | -0.0141 | **-0.0475** (0/15, t = -124) |
| per-country marginal-quantile calibration to the pooled target rank | -0.0084 (0/15, t = -12.1) | **-0.0267** (0/15, t = -27.3) |

The cost of getting this wrong grew five-fold. Every blend below is z-scored **pooled**.

What *did* change is where the optimum sits. Sweeping an oracle rescaling of the DE block
(FR fixed at 1, applied to per-country z-scores):

| s_DE | old regime | time-ordered regime |
|---|---|---|
| 1.0 | -0.0094 | -0.0484 |
| 2.0 | -0.0003 | -0.0097 |
| 3.0 | -0.0009 | +0.0009 |
| 5.0 | - | +0.0056 |
| **6.0** | - | **+0.0059** (15/15, t = +14.3) |
| 10.0 | - | +0.0050 |
| raw prediction ratio DE/FR | 2.14 (= the optimum) | 2.86 (below the optimum) |

In the old regime the raw scale *was* the optimum, so there was nothing to gain. Now the
oracle wants DE stretched about twice as wide as the models produce, worth up to +0.0059.
Fitting that single scale parameter honestly inside the fold recovers **+0.0021 +/- 0.0047
(12/15, t = +1.75)** - the right sign, but the grid choice is noisy and the result is not
significant. Flagged as the one piece of unclaimed headroom found in this study.

### 1b. Cross-country transfer - went from a loss to a no-op

Giving each country's ridge the other country's (nested, inner-OOF) prediction as an extra
feature. FR and DE rows of a day are bit-identical in the features, so this is legitimate.

| variant | old regime (base 0.2919) | time-ordered (base 0.5191) |
|---|---|---|
| FR model gets the DE model's prediction | -0.0029 (0/15, t = -7.3) | -0.0001 (5/15, t = -1.4) |
| both countries get the other's prediction | -0.0029 (0/15, t = -8.1) | -0.0002 (4/15, t = -1.9) |

Under thin signal the extra column was pure added variance. With the time features the FR
model already has what the DE model could tell it, so the transfer is neither harmful nor
useful. Conclusion unchanged in practice: **do not do it.**

