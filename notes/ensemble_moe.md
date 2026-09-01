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

