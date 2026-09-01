# Model zoo, re-run on the ID-ordered time features

Everything in `notes/model_zoo.md` was measured in the 0.29 regime. The `DAY_ID`-decoy
finding moved the target to ~0.52, and the coordinator was right to be suspicious: **the
headline conclusion of that document inverts, and five of its six confirmed single-model
findings turn negative.** Its ensembling conclusion survives in direction but not in
composition. This file supersedes it.

Sections 1-9 are measured against the 0.5231 time-feature baseline. **Section 10 re-tests
the headline claims against the newer `add_cumulative_returns` commit (83aa235), and one
of my own findings dies there -- read it before acting on anything above it.**

## TL;DR

1. **"Added capacity always loses" is dead.** The *identical* LightGBM configuration that
   lost by 0.0125 on the base features now **beats ridge by +0.0232 (15/15 seeds)**,
   scoring 0.5463 against 0.5231. HistGradientBoosting flips the same way, -0.0274 ->
   +0.0114. The rule was never about model class; it was about signal-to-noise. With 58
   informative time features there is finally enough signal to support interactions.
2. **The diversity story inverts with it.** In the 0.29 regime, boosting was worthless
   alone and valuable only as a blend partner. Now it is the best standalone model, and
   *ridge* is the member you drop. Best found:
   **`0.40*z(LightGBM) + 0.40*z(ARD) + 0.20*z(pooled LightGBM)`, no ridge at all --
   0.5617 on seeds 0..14, confirmed at 0.5597 on 30 unseen seeds (+0.0352, sd 0.0049,
   30/30)**. `LGBM + ARD` alone gets most of it (+0.0332, 30/30).
3. **One of my own findings did not survive the next feature commit -- see section 10.**
   `ARDRegression`'s +0.0164 collapses to **+0.0003 (10/15)** once the cumulative
   commodity-return columns land, because ARD was compensating for a missing France
   feature rather than modelling France better. Point 5, and the France half of point 4,
   are withdrawn on that basis. LightGBM's win survives the same test at +0.0187 (15/15),
   and the `LGBM + ARD` blend survives at +0.0233 with a much reduced margin.
4. **The best *single* model on the 0.5231 feature set is a per-country split: ARD for
   France, LightGBM for Germany.** 0.5542 on seeds 0..14 (+0.0311, 15/15), **confirmed at 0.5519 on 30 unseen
   seeds (+0.0275, sd 0.0049, 30/30)**. It captures 83% of the blend's gain in one
   estimator that fits in 11 seconds for all 15 seeds. The per-country asymmetry survives
   the regime change -- it just points the other way now, at *which family* each country
   wants rather than at how much capacity it tolerates.
5. **The biggest surprise is a linear model (but see point 3).** `ARDRegression`, worth +0.0009 (noise) on
   29 features, is worth **+0.0164 (15/15)** on 87. Automatic relevance determination is
   doing feature selection that ridge's single global alpha cannot: most of the 58 new
   columns are noise, and ARD prunes them per-coefficient. Sparsity was harmful in the old
   regime and is essential in this one -- an exact reversal.
6. **What did NOT transfer, from my own prior document.** `Ridge-std` +0.0013 -> **-0.0040
   (0/15)**; `Huber` +0.0020 -> **-0.0273 (0/15)**; `PLS` +0.0014 -> **-0.0275 (0/15)**;
   `C1` FR-ridge/DE-spline +0.0035 -> **-0.0016 (5/15)**; `C2` FR-ridge/DE-KRR +0.0038 ->
   **-0.0109 (0/15)**. The *specific* recipe those hybrids encoded -- "France cannot take
   nonlinearity, Germany can take a little" -- is dead; the asymmetry itself survives, but
   as point 4 (France wanted ARD, Germany wants trees). Only `Ridge-winsor` survives
   unchanged, and weakly: +0.0022 -> +0.0021 (11/15).
7. **Kernel and instance methods got much worse, not better.** `SVR-rbf` -0.0137 ->
   -0.0538; `kNN-std-dist` -0.0379 -> -0.0829. They cannot cope with 87 columns of mixed
   scale where most are noise. Their OOF correlation with ridge fell (0.85 -> 0.78,
   0.81 -> 0.76), but that is now the signature of a broken model rather than a useful one.
8. **Tuning boosting makes it worse; the architecture is the whole effect.** Inner-CV-tuned
   LightGBM lands at +0.0105 against the untuned conservative config's +0.0232, and tuned
   HistGB at +0.0054 against +0.0114. I pre-registered a prediction in section 7 that tuned
   LightGBM would land in +0.023..+0.035; it came in below the range *and* below the fixed
   config, so that prediction was wrong. Use a fixed conservative booster, not a search --
   point 1 never depended on a tuning budget, which makes it safer rather than shakier.

## Protocol

* **Features.** `TO.add_time_features(train, test, F, lags=(1,), windows=(7,))` -> 58 new
  columns (first difference at lag 1, and value-minus-trailing-7-day-mean, for each of the
  29 base features). FR uses `SEL + NEW` (61 columns), DE uses `F + NEW` (87).
* **Baselines.** Per-country `RidgeCV(alphas=np.logspace(-2,4,40))` on median-imputed raw
  features with a within-country rank-transformed target:
  * **time-ridge = 0.5231 +/- 0.0061** (15 seeds) -- the paired reference for every row
    below, since every candidate uses the same feature set.
  * **time-ridge + neighbour(k=3) = 0.5277 +/- 0.0076** -- reproduces the coordinator's
    0.5282. Used as the reference for section 6 only.
* **Pairing.** Identical `P.make_folds(train.DAY_ID, seed=s)` assignments, seeds 0..14.
  `delta` is the mean of the 15 per-seed differences, `sd(delta)` their sd, `wins/15` the
  count of seeds where the candidate is higher.
* **Leakage.** Time features are built from X only (train and test), with `.shift(1)` inside
  every rolling window, so they are computable at submission time and contain no target;
  they are therefore built once, outside the fold loop. Everything target-dependent --
  imputation medians, scaling, winsorisation cut-points, the rank transform, and all
  hyperparameter selection -- is fitted inside the fold on that country's training rows.
  The neighbour-target features in section 6 are the one genuinely leak-prone piece and are
  rebuilt per fold with sources restricted to that fold's training rows of that country,
  with the identity match dropped.
* **Sanity check on the jump.** 0.29 -> 0.52 is large enough to deserve suspicion. It is not
  leakage: the added columns are differences and rolling deviations of X, the fold split is
  still grouped on DAY_ID, and the ridge that produces 0.5231 is the same estimator that
  produces 0.2909 on the base features. The gain comes from the features, and the
  organisers supply test X, so it survives to submission.
* **One transductive detail, named rather than hidden.** `add_time_features` clips the
  derived columns at 8 sd, and that sd is computed over the whole train+test derived frame,
  i.e. outside the fold. It is a feature-only statistic with no target in it, so it cannot
  leak label information, and it is computable at submission time since test X is supplied.
  It is still a global constant rather than a fold-local one; anyone who wants the estimate
  to be airtight can move it inside the fold, at the cost of clipping test rows with a
  train-only sd.

## 1. Full sweep on time features, paired against time-ridge (seeds 0..14)

| model | config | score | delta vs ridge | sd(delta) | wins/15 | corr(ridge) | FR rho | DE rho |
|---|---|---|---|---|---|---|---|---|
| `FRard-DElgbm` | ARD for France, LightGBM for Germany, both on z-scored features | 0.5542 | +0.0311 | 0.0040 | 15/15 | 0.913 | 0.2666 | 0.7850 |
| `LGBM-leaves4` | identical config to the 0.29-regime run | 0.5463 | +0.0232 | 0.0072 | 15/15 | 0.881 | 0.2426 | 0.7847 |
| `LGBM-stdprep` | LightGBM on z-scored features (control: trees are scale-invariant) | 0.5461 | +0.0230 | 0.0077 | 15/15 | 0.881 | 0.2419 | 0.7850 |
| `ARD-std-check` | determinism control: ARDRegression re-run in a separate process, must match the row above | 0.5395 | +0.0164 | 0.0042 | 15/15 | 0.926 | 0.2666 | 0.7677 |
| `ARDRegression` | default (was +0.0009, best DE linear) | 0.5395 | +0.0164 | 0.0042 | 15/15 | 0.926 | 0.2666 | 0.7677 |
| `CatBoost-fixed` | depth 4, lr .03, 400 iters, l2 5 (no tuning) | 0.5387 | +0.0156 | 0.0063 | 15/15 | 0.866 | 0.2320 | 0.7808 |
| `POOLED-CatBoost-fixed` | pooled over FR+DE with COUNTRY, depth 6, 600 iters | 0.5358 | +0.0127 | 0.0083 | 14/15 | 0.837 | 0.2209 | 0.7723 |
| `HGB-shallow` | identical config to the 0.29-regime run | 0.5345 | +0.0114 | 0.0070 | 14/15 | 0.856 | 0.2329 | 0.7787 |
| `LGBM-tuned` | num_leaves tuned inner (4/15/31), 250 rounds, lr .03, colsample .6 | 0.5336 | +0.0105 | 0.0088 | 13/15 | 0.850 | 0.2308 | 0.7802 |
| `Spline+Ridge` | cubic B-splines per feature, n_knots tuned inner, RidgeCV head | 0.5286 | +0.0055 | 0.0052 | 13/15 | 0.934 | 0.2251 | 0.7684 |
| `HGB-tuned` | max_depth tuned inner (2/4/None), 250 iters, lr .03, l2 5 | 0.5285 | +0.0054 | 0.0082 | 10/15 | 0.830 | 0.2249 | 0.7787 |
| `Ridge-winsor` | features clipped to fold-train 2/98 pct (was +0.0022, 25/30) | 0.5251 | +0.0021 | 0.0028 | 11/15 | 0.969 | 0.2137 | 0.7718 |
| `C1-FRridge-DEspline` | ridge FR + splines DE (was +0.0035, 22/30) | 0.5215 | -0.0016 | 0.0032 | 5/15 | 0.963 | 0.2053 | 0.7684 |
| `POOLED-LGBM-leaves15` | pooled, COUNTRY feature, leaves 15 / 600 rounds (2x the rows) | 0.5214 | -0.0017 | 0.0077 | 6/15 | 0.803 | 0.2122 | 0.7614 |
| `Ridge-std` | RidgeCV on z-scored features (was +0.0013, 27/30) | 0.5191 | -0.0040 | 0.0029 | 0/15 | 0.976 | 0.2053 | 0.7690 |
| `C2-FRridge-DEkrr` | ridge FR + kernel ridge DE (was +0.0038, 25/30) | 0.5122 | -0.0109 | 0.0034 | 0/15 | 0.966 | 0.2053 | 0.7581 |
| `Huber` | alpha tuned inner (was +0.0020, 10/15) | 0.4958 | -0.0273 | 0.0065 | 0/15 | 0.928 | 0.1953 | 0.7600 |
| `PLS` | n_components tuned inner (was +0.0014, 9/15) | 0.4956 | -0.0275 | 0.0061 | 0/15 | 0.921 | 0.1929 | 0.7597 |
| `POOLED-HGB-shallow` | pooled, COUNTRY feature, HistGB depth 2 | 0.4864 | -0.0367 | 0.0076 | 0/15 | 0.770 | 0.1505 | 0.7303 |
| `POOLED-LGBM-leaves4` | pooled, COUNTRY feature, same leaves-4 config as the per-country winner | 0.4738 | -0.0493 | 0.0070 | 0/15 | 0.739 | 0.1340 | 0.7429 |
| `SVR-rbf` | C/gamma tuned inner, identical grid | 0.4693 | -0.0538 | 0.0108 | 0/15 | 0.777 | 0.2084 | 0.7278 |
| `kNN-std-dist` | z-scored feats, distance weights, k tuned | 0.4402 | -0.0829 | 0.0097 | 0/15 | 0.761 | 0.1721 | 0.7115 |
| `POOLED-ridge` | one RidgeCV over FR+DE with a COUNTRY indicator (sanity check) | 0.3951 | -0.1280 | 0.0044 | 0/15 | 0.702 | 0.1339 | 0.6820 |


## 2. Diversity map: pairwise OOF Spearman between models

Averaged over the 15 seeds. Compare with the 0.29 regime, where the same
boosting/SVR/kNN members sat at 0.78-0.85 against ridge.

| | `ridge` | `ARDRegression` | `LGBM-leaves4` | `HGB-shallow` | `CatBoost-fixed` | `Spline+Ridge` | `Ridge-winsor` | `FRard-DElgbm` | `POOLED-LGBM-leaves15` | `SVR-rbf` | `kNN-std-dist` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ridge` | 1.000 | 0.926 | 0.881 | 0.856 | 0.866 | 0.934 | 0.969 | 0.913 | 0.803 | 0.777 | 0.761 |
| `ARDRegression` | 0.926 | 1.000 | 0.858 | 0.834 | 0.843 | 0.882 | 0.910 | 0.970 | 0.781 | 0.747 | 0.715 |
| `LGBM-leaves4` | 0.881 | 0.858 | 1.000 | 0.962 | 0.958 | 0.926 | 0.908 | 0.896 | 0.887 | 0.771 | 0.764 |
| `HGB-shallow` | 0.856 | 0.834 | 0.962 | 1.000 | 0.950 | 0.899 | 0.879 | 0.870 | 0.871 | 0.757 | 0.739 |
| `CatBoost-fixed` | 0.866 | 0.843 | 0.958 | 0.950 | 1.000 | 0.917 | 0.890 | 0.879 | 0.887 | 0.793 | 0.761 |
| `Spline+Ridge` | 0.934 | 0.882 | 0.926 | 0.899 | 0.917 | 1.000 | 0.961 | 0.891 | 0.851 | 0.803 | 0.794 |
| `Ridge-winsor` | 0.969 | 0.910 | 0.908 | 0.879 | 0.890 | 0.961 | 1.000 | 0.904 | 0.824 | 0.785 | 0.779 |
| `FRard-DElgbm` | 0.913 | 0.970 | 0.896 | 0.870 | 0.879 | 0.891 | 0.904 | 1.000 | 0.814 | 0.748 | 0.726 |
| `POOLED-LGBM-leaves15` | 0.803 | 0.781 | 0.887 | 0.871 | 0.887 | 0.851 | 0.824 | 0.814 | 1.000 | 0.760 | 0.714 |
| `SVR-rbf` | 0.777 | 0.747 | 0.771 | 0.757 | 0.793 | 0.803 | 0.785 | 0.748 | 0.760 | 1.000 | 0.691 |
| `kNN-std-dist` | 0.761 | 0.715 | 0.764 | 0.739 | 0.761 | 0.794 | 0.779 | 0.726 | 0.714 | 0.691 | 1.000 |

## 3. Blends with time-ridge (pooled-z, seeds 0..14)

`(1-w)*z(ridge) + w*z(candidate)`. Weights are scanned, so treat as an upper bound.

| candidate | corr(ridge) | w=0.25 | delta | w=0.40 | delta | w=0.50 | delta |
|---|---|---|---|---|---|---|---|
| `FRard-DElgbm` | 0.913 | 0.5395 | +0.0165 (15/15) | 0.5462 | +0.0231 (15/15) | 0.5493 | +0.0262 (15/15) |
| `LGBM-leaves4` | 0.881 | 0.5402 | +0.0172 (15/15) | 0.5463 | +0.0232 (15/15) | 0.5487 | +0.0257 (15/15) |
| `LGBM-stdprep` | 0.881 | 0.5402 | +0.0171 (15/15) | 0.5462 | +0.0231 (15/15) | 0.5486 | +0.0255 (15/15) |
| `POOLED-CatBoost-fixed` | 0.837 | 0.5409 | +0.0178 (15/15) | 0.5462 | +0.0232 (15/15) | 0.5478 | +0.0247 (15/15) |
| `CatBoost-fixed` | 0.866 | 0.5385 | +0.0154 (15/15) | 0.5436 | +0.0205 (15/15) | 0.5453 | +0.0222 (15/15) |
| `LGBM-tuned` | 0.850 | 0.5389 | +0.0158 (15/15) | 0.5434 | +0.0203 (15/15) | 0.5445 | +0.0214 (15/15) |
| `HGB-shallow` | 0.856 | 0.5382 | +0.0151 (15/15) | 0.5426 | +0.0195 (15/15) | 0.5439 | +0.0208 (15/15) |
| `POOLED-LGBM-leaves15` | 0.803 | 0.5403 | +0.0172 (15/15) | 0.5439 | +0.0208 (15/15) | 0.5437 | +0.0206 (15/15) |
| `HGB-tuned` | 0.830 | 0.5390 | +0.0159 (15/15) | 0.5426 | +0.0196 (15/15) | 0.5431 | +0.0200 (15/15) |
| `ARDRegression` | 0.926 | 0.5328 | +0.0097 (15/15) | 0.5368 | +0.0137 (15/15) | 0.5387 | +0.0156 (15/15) |
| `ARD-std-check` | 0.926 | 0.5328 | +0.0097 (15/15) | 0.5368 | +0.0137 (15/15) | 0.5387 | +0.0156 (15/15) |
| `POOLED-HGB-shallow` | 0.770 | 0.5338 | +0.0107 (15/15) | 0.5314 | +0.0083 (15/15) | 0.5270 | +0.0039 (13/15) |
| `POOLED-LGBM-leaves4` | 0.739 | 0.5327 | +0.0096 (15/15) | 0.5279 | +0.0049 (14/15) | 0.5218 | -0.0012 (7/15) |
| `Spline+Ridge` | 0.934 | 0.5291 | +0.0060 (15/15) | 0.5313 | +0.0082 (15/15) | 0.5321 | +0.0090 (15/15) |
| `SVR-rbf` | 0.777 | 0.5288 | +0.0057 (15/15) | 0.5250 | +0.0019 (10/15) | 0.5199 | -0.0032 (3/15) |
| `Ridge-winsor` | 0.969 | 0.5256 | +0.0025 (15/15) | 0.5266 | +0.0035 (15/15) | 0.5270 | +0.0039 (15/15) |
| `C1-FRridge-DEspline` | 0.963 | 0.5257 | +0.0026 (15/15) | 0.5263 | +0.0032 (15/15) | 0.5262 | +0.0031 (14/15) |
| `Ridge-std` | 0.976 | 0.5237 | +0.0006 (12/15) | 0.5237 | +0.0006 (11/15) | 0.5233 | +0.0003 (11/15) |
| `C2-FRridge-DEkrr` | 0.966 | 0.5229 | -0.0001 (8/15) | 0.5221 | -0.0010 (3/15) | 0.5211 | -0.0019 (1/15) |
| `PLS` | 0.921 | 0.5216 | -0.0015 (3/15) | 0.5186 | -0.0044 (2/15) | 0.5159 | -0.0072 (0/15) |
| `Huber` | 0.928 | 0.5212 | -0.0019 (1/15) | 0.5181 | -0.0050 (0/15) | 0.5155 | -0.0076 (0/15) |
| `kNN-std-dist` | 0.761 | 0.5210 | -0.0021 (3/15) | 0.5132 | -0.0099 (0/15) | 0.5053 | -0.0177 (0/15) |
| `POOLED-ridge` | 0.702 | 0.5134 | -0.0097 (0/15) | 0.4971 | -0.0260 (0/15) | 0.4834 | -0.0397 (0/15) |


## 4. Ensembles (seeds 0..14)

Members and weights are chosen on these seeds, so treat as an upper bound;
section 5 re-scores the survivors on fold seeds never used for selection.

| ensemble (pooled-z weights) | score | delta vs time-ridge | sd(delta) | wins/15 |
|---|---|---|---|---|
| ridge alone (reference) | 0.5231 | +0.0000 | 0.0000 | 0/15 |
| LGBM alone | 0.5463 | +0.0232 | 0.0072 | 15/15 |
| 0.60 ridge + 0.40 LGBM  (the 0.29-regime recipe) | 0.5463 | +0.0232 | 0.0030 | 15/15 |
| 0.50 ridge + 0.25 LGBM + 0.25 SVR (the 0.29-regime best) | 0.5434 | +0.0203 | 0.0032 | 15/15 |
| 0.50 ridge + 0.50 LGBM | 0.5487 | +0.0257 | 0.0038 | 15/15 |
| equal ridge + LGBM + ARD | 0.5539 | +0.0308 | 0.0030 | 15/15 |
| equal ridge + LGBM + ARD + HGB | 0.5550 | +0.0319 | 0.0039 | 15/15 |
| 0.50 LGBM + 0.50 ARD  (no ridge) | 0.5593 | +0.0362 | 0.0044 | 15/15 |
| equal LGBM + ARD + HGB  (no ridge) | 0.5569 | +0.0338 | 0.0050 | 15/15 |
| equal LGBM + ARD + Spline  (no ridge) | 0.5554 | +0.0323 | 0.0040 | 15/15 |
| 0.40 LGBM + 0.40 ARD + 0.20 pooled-LGBM  (BEST) | 0.5617 | +0.0386 | 0.0048 | 15/15 |
| 0.40 LGBM + 0.40 ARD + 0.20 pooled-CatBoost | 0.5619 | +0.0388 | 0.0048 | 15/15 |
| ARD-FR/LGBM-DE single estimator (no blend) | 0.5542 | +0.0311 | 0.0040 | 15/15 |

## 5. Confirmation on 30 fresh fold seeds (15..44)

Members and weights fixed before running.

fresh seeds 15..44 (n=30)   time-ridge 0.5245 +/- 0.0082

| configuration (fixed in advance) | score | delta vs time-ridge | sd(delta) | wins/30 |
|---|---|---|---|---|
| ridge | 0.5245 | +0.0000 | 0.0000 | 0/30 |
| ARDRegression alone | 0.5376 | +0.0131 | 0.0052 | 30/30 |
| LGBM alone | 0.5452 | +0.0207 | 0.0068 | 30/30 |
| HGB alone | 0.5309 | +0.0065 | 0.0065 | 26/30 |
| ARD-FR + LightGBM-DE (single model) | 0.5519 | +0.0275 | 0.0049 | 30/30 |
| 0.50 LGBM + 0.50 ARD (no ridge) | 0.5577 | +0.0332 | 0.0044 | 30/30 |
| 0.40 LGBM + 0.40 ARD + 0.20 pooled-LGBM | 0.5597 | +0.0352 | 0.0049 | 30/30 |
| 0.50 (ARD-FR/LGBM-DE) + 0.50 ARD | 0.5504 | +0.0259 | 0.0048 | 30/30 |
| equal LGBM + ARD + HGB (no ridge) | 0.5543 | +0.0298 | 0.0050 | 30/30 |
| equal ridge + LGBM + ARD + HGB | 0.5532 | +0.0287 | 0.0037 | 30/30 |

## 6. Temporal-neighbour features

Scored against the ridge+neighbour(k=3) baseline of 0.5277, not plain time-ridge.

| model | score | delta vs ridge+nb | sd(delta) | wins/15 | corr(ridge+nb) |
|---|---|---|---|---|---|
| `ridge + neighbour(k=3)` | 0.5277 | +0.0046 vs plain time-ridge | 0.0053 | 13/15 | 1.000 |
| `NB-HGB-shallow` | 0.5368 | +0.0091 | 0.0087 | 12/15 | 0.849 |
| `NB-LGBM-leaves4` | 0.5486 | +0.0209 | 0.0077 | 15/15 | 0.874 |

## 7. What changed, and why

**The regime change is a signal-to-noise change, not a model-class change.** On the base
features, ridge's pooled OOF was 0.29 and the honest spread between the best and worst
sensible model was ~0.08. On the time features ridge is 0.52 and the spread is ~0.11, but
the *ordering* has flipped: the families that were penalised for capacity are now rewarded
for it. Two mechanisms, both visible in the tables:

1. *There is now enough signal to identify interactions.* 58 of the 87 DE columns are
   first differences and deviations-from-trailing-mean. Electricity price moves are driven
   by things like "residual load jumped **and** gas is above its recent regime", which is a
   product term a linear model cannot represent. With 0.29 of signal, estimating those
   interactions cost more variance than they bought. With 0.52, they pay: LightGBM
   -0.0125 -> +0.0232, HistGB -0.0274 -> +0.0114, splines -0.0009 -> +0.0055.
2. *There are now enough columns for relevance weighting to matter.* Going from 29 to 87
   features means most columns are noise. Ridge's single global alpha must shrink the good
   ones to control the bad ones; `ARDRegression`'s per-coefficient prior does not. That is
   worth +0.0164 with no nonlinearity at all, and it is the single cleanest reversal in the
   sweep: sparsity was *harmful* on 29 features and is *essential* on 87.

**The whole 0.29 -> 0.52 jump is Germany, and France is still the open problem.**
Per country, ridge goes FR 0.2084 -> 0.2265 (+0.018) and DE 0.3598 -> 0.7630 (+0.403).
The recovered ordering essentially solves Germany and barely touches France. That makes
the France column the place to look, and there is a large result sitting in it:
**`ARDRegression` scores FR rho 0.2666 against time-ridge's 0.2265, +0.040** -- by a wide
margin the largest France gain in either document, and more than double what the time
features themselves bought France. LightGBM manages FR 0.2426. France has 61 columns and
851 rows, of which 58 columns are derived and mostly irrelevant to it; ridge's single alpha
has to shrink everything to survive that, and ARD does not. Anyone working on France next
should start from ARD, not ridge. Acting on that directly -- ARD for France, LightGBM
for Germany, one estimator per country as usual -- gives **0.5542 (+0.0311, 15/15),
confirmed at +0.0275 (30/30) on fresh seeds**, which is 83% of what the full blend buys
for a model that fits in 11 seconds.

**Tuning boosting makes it WORSE here, and my prediction about that was wrong.** Both
inner-CV-tuned grids finished after the rest of this document was written, and both lost
to the hand-set conservative configs they were meant to improve on:

| model | tuned | fixed | fixed config |
|---|---|---|---|
| LightGBM | +0.0105 (13/15) | **+0.0232 (15/15)** | 4 leaves, 300 rounds, lr .02, l2 20 |
| HistGB | +0.0054 (10/15) | **+0.0114 (14/15)** | depth 2, 300 iters, lr .03, l2 10 |

I had recorded a prediction in this file before the runs landed -- "tuned LightGBM lands
between +0.023 and +0.035" -- and it came in at +0.0105, below the whole range and below
the untuned config. Selecting `num_leaves` from {4, 15, 31} by 3-fold inner CV on ~515
German rows is itself a noisy operation, and it kept choosing deeper trees than the fixed
choice. So the mechanism is architecture, not tuning: gradient boosting on differenced
time features beats ridge, and the way to get that benefit is a *fixed conservative*
config, not a search. That also makes the headline safer rather than shakier -- the
+0.0232 result never depended on a tuning budget.

The fixed configs used for the headline rows are, if anything, still under-powered: they
are the exact settings chosen to cripple capacity in the 0.29 regime (4 leaves, depth 2,
`reg_lambda=20`, `min_child_samples=40`, 300 rounds at lr 0.02). `CatBoost-fixed` at
depth 4 -- more capacity, no tuning -- lands at +0.0156, between the two.

Every row in this document did eventually complete, but only just: the machine spent most
of the session at load 190-230 with a system-wide OOM that killed every python process at
one point, and the last row (the pooled CatBoost, 1474 s) landed after the conclusions had
already been written -- and changed one of them. See the pooled paragraph below.

**The three boosting libraries are one member, not three.** LightGBM, HistGradientBoosting
and CatBoost sit at 0.95-0.96 mutual OOF correlation and score 0.5463 / 0.5345 / 0.5387.
Choose on speed, not on diversity: LightGBM is both the strongest and the fastest here
(83 s for all 15 seeds against CatBoost's 508 s single-threaded).

**Boosting and ARD are complementary, and ridge is redundant to both.** LightGBM's OOF
correlates 0.881 with ridge and 0.858 with ARD; ARD correlates 0.926 with ridge. So the
tree model contributes nonlinearity, ARD contributes a better-selected linear part, and
ridge contributes a strictly worse version of what ARD already provides. Dropping ridge
from the ensemble *improves* it: `LGBM + ARD` at 0.5593 beats `ridge + LGBM + ARD` at
0.5539 and `ridge + LGBM + ARD + HGB` at 0.5550. On fresh seeds, `0.5 LGBM + 0.5 ARD`
is +0.0332 (30/30) against time-ridge's +0.0000.

**The winner's pooled-with-COUNTRY shape works, but only with real capacity -- and I got
this wrong the first time.** My first four pooled runs all lost, and I had written the
result up as a clean negative. The fifth, a depth-6 CatBoost with 600 rounds, finished last
and **beats ridge by +0.0127 (14/15)**. The pattern across all five is capacity, not
pooling:

| pooled model (COUNTRY as a feature) | delta vs ridge | wins |
|---|---|---|
| RidgeCV | -0.1280 | 0/15 |
| LightGBM, 4 leaves (per-country winner's settings) | -0.0493 | 0/15 |
| HistGB, depth 2 | -0.0367 | 0/15 |
| LightGBM, 15 leaves / 600 rounds | -0.0017 | 6/15 |
| **CatBoost, depth 6 / 600 rounds** | **+0.0127** | **14/15** |

That is exactly the ordering you would predict once you accept point 1: pooling doubles the
rows, and doubling the rows *raises* the capacity the data will support, so a pooled model
needs to be bigger than the per-country one, not the same size. Transplanting the
per-country settings is what made it look dead. The pooled ridge row stays at -0.1280 and
is still informative: a country dummy can only move the intercept, so a linear pooled model
genuinely cannot work no matter how it is fitted.

The corrected conclusion is narrower than "pooling loses". Per-country still wins on raw
score -- +0.0232 for per-country LightGBM against +0.0127 for the best pooled model -- so
I would not submit a pooled model alone. But pooling is a real and *different* view: the
pooled CatBoost correlates 0.837 with ridge and only 0.819 with ARD, and as a 20% ensemble
member it is worth +0.0388 against the two-member blend's +0.0362. The pooled LightGBM
gives +0.0386 for a fifth of the fit time, so either serves; they are 0.93 correlated with
each other and count as one slot. If the challenge winner's CatBoost really was pooled, a
depth-6-class model is the shape that makes that plausible.

**Everything that made kNN and SVR interesting is gone.** Both were mediocre-but-diverse in
the 0.29 regime and were worth +0.004 to +0.006 in a blend. On 87 columns they are simply
broken: SVR-rbf -0.0538, kNN -0.0829. Their correlation with ridge fell (0.85 -> 0.78,
0.81 -> 0.76), which in the old document was the marker of a useful ensemble member and
here is just the signature of a model that has lost the plot. Low correlation is only
worth having when the model is still tracking the target.

**My old blend recipe survives in direction but not in composition.** The 0.29-regime
best -- `0.50 ridge + 0.25 LightGBM + 0.25 SVR-rbf` -- still gains here, +0.0203 (15/15),
because the LightGBM half is now genuinely strong. But it is beaten by LightGBM on its own
(+0.0232) and comfortably beaten by `LGBM + ARD` (+0.0362), because a quarter of its weight
is on SVR-rbf, which is now the second-worst model in the sweep. The general lesson
transfers; the specific recipe does not.

**Neighbour-target features and boosting overlap but do not cancel.** Ridge gains +0.0046
from the k=3 neighbour features (0.5231 -> 0.5277). LightGBM gains only +0.0023 from them
(0.5463 -> 0.5486), and stays +0.0209 ahead of ridge+neighbour. Both are reading the same
recovered ordering, so roughly half of what the neighbour features give a linear model is
already available to a tree through the differenced columns.

## 8. Exact constructors

```python
import sys; sys.path.insert(0, "src")
import numpy as np, qrt_prep as P, qrt_timeorder as TO
from scipy.stats import rankdata

train, test = P.build_frames()
F   = P.feature_columns(train)
SEL = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"]
tr, te, NEW = TO.add_time_features(train, test, F, lags=(1,), windows=(7,))
CF, CD = SEL + NEW, F + NEW                    # FR 61 cols, DE 87 cols

# per fold, per country: median-impute from the fold's training rows, rank the target
med = t[cols].median()
Xtr = np.nan_to_num(t[cols].fillna(med).to_numpy(float))
Xva = np.nan_to_num(v[cols].fillna(med).to_numpy(float))
ytr = rankdata(t.TARGET.values) / len(t)
```

### 8.1 Best single family: LightGBM, per country (+0.0207, 30/30 on fresh seeds)

Raw (unscaled) imputed features. This is byte-for-byte the config that *lost* by 0.0125
on the base features -- nothing about it was retuned for the new regime.

```python
import lightgbm as lgb
lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=4,
                  min_child_samples=40, subsample=0.7, subsample_freq=1,
                  colsample_bytree=(1.0 if country == "FR" else 0.4),
                  reg_lambda=20.0, random_state=seed, n_jobs=1, verbose=-1)
```

### 8.2 Best linear model: ARDRegression, per country (+0.0131, 30/30 on fresh seeds)

Needs z-scored features (`prep="std"`); it is scale-sensitive in a way RidgeCV is not.
No hyperparameters, no inner CV, 2 seconds for all 15 seeds.

```python
from sklearn.linear_model import ARDRegression
ARDRegression()                      # on (X - mu) / sd, fitted inside the fold
```

### 8.3 Best single estimator: ARD for France, LightGBM for Germany (+0.0275, 30/30 fresh)

Both heads on `prep="std"` -- ARD needs it and trees are indifferent (control run
`LGBM-stdprep` scores 0.5461 against 0.5463 raw, so the shared prep costs nothing).

```python
def make_model(country, seed):
    if country == "FR":
        return ARDRegression()
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=4,
                             min_child_samples=40, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.4, reg_lambda=20.0,
                             random_state=seed, n_jobs=1, verbose=-1)
```

### 8.4 BEST OVERALL: drop ridge (+0.0352, 30/30 on fresh seeds)

Third member is one LightGBM fitted over FR+DE together with a COUNTRY indicator column
appended to the 87 features and the target ranked over the pooled fold-training rows. It
scores at ridge's level on its own; it earns its place by being uncorrelated with the
per-country models (0.803 with ridge, 0.887 with per-country LightGBM).
`POOLED-CatBoost-fixed` (depth 6, 600 iters) is interchangeable in that slot: better
alone (+0.0127 vs -0.0017) and +0.0388 in the blend against +0.0386, but 1474 s of fit
time against 156 s. The two are 0.93 correlated -- use one, not both.

```python
z = lambda v: (v - v.mean()) / v.std()          # global, never per country
pred = 0.40 * z(lgbm_pred) + 0.40 * z(ard_pred) + 0.20 * z(pooled_lgbm_pred)
# 0.5617 on seeds 0..14 ; 0.5597 on 30 unseen fold seeds
# drop the third member and it is 0.5593 / 0.5577 -- still worth having

lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, reg_lambda=5.0,
                  random_state=seed, n_jobs=1, verbose=-1)   # the pooled member
```

## 9. Out-of-fold bundle

`notes/oof_zoo_time.pkl` is a `dict[str, list[np.ndarray]]`, 15 OOF vectors per model
(fold seeds 0..14, in order), each of length 1494 aligned to `P.build_frames()[0]` row
order. Members: `ridge_time`, `ridge_time_nb`, `LGBM-leaves4`, `ARDRegression`,
`HGB-shallow`, `CatBoost-fixed`, `Spline+Ridge`, `SVR-rbf`, `kNN-std-dist`,
`Ridge-winsor`, `FRard-DElgbm`, `POOLED-LGBM-leaves15`, `POOLED-CatBoost-fixed`,
`LGBM-tuned`, `HGB-tuned`, `NB-LGBM-leaves4`, `NB-HGB-shallow` -- 17 in all.

```python
import pickle, numpy as np
from scipy.stats import spearmanr
oof = pickle.load(open("notes/oof_zoo_time.pkl", "rb"))
z = lambda v: (v - v.mean()) / v.std()
best = [0.4 * z(l) + 0.4 * z(a) + 0.2 * z(p)
        for l, a, p in zip(oof["LGBM-leaves4"], oof["ARDRegression"],
                           oof["POOLED-LGBM-leaves15"])]              # 0.5617 on seeds 0..14
```

`SVR-rbf` and `kNN-std-dist` are kept in the bundle only so the negative result is
reproducible; do not put them in an ensemble.

## 10. Re-test against commit 83aa235 (cumulative commodity returns) -- ONE FINDING DIES

While this sweep was running, another agent landed `add_cumulative_returns`, which fixes
the France bottleneck with a feature (trailing sums of GAS/COAL/CARBON returns, 6 columns).
Since section 7 argues France is the open problem and ARD is the answer to it, that commit
is a direct test of my own conclusion. Same harness, same folds, same pairing, with the 6
cumulative columns appended to both countries (FR 67 cols, DE 93):

| model | score | delta vs ridge | sd(delta) | wins/15 | FR rho | DE rho |
|---|---|---|---|---|---|---|
| `ridge` (new reference) | 0.5439 | +0.0000 | 0.0000 | 0/15 | 0.2874 | 0.7595 |
| `ARDRegression` | 0.5442 | **+0.0003** | 0.0034 | 10/15 | 0.2821 | 0.7681 |
| `LGBM-leaves4` | 0.5626 | **+0.0187** | 0.0070 | 15/15 | 0.2896 | 0.7838 |
| `ARD-FR + LGBM-DE` | 0.5584 | +0.0145 | 0.0036 | 15/15 | 0.2821 | 0.7836 |
| `0.5 LGBM + 0.5 ARD` | 0.5672 | +0.0233 | 0.0037 | 15/15 | | |

**The ARD finding does not survive.** +0.0164 (15/15) becomes +0.0003 (10/15) -- pure
noise. And the mechanism is now obvious in hindsight: ARD was never solving France, it was
*compensating* for a missing France feature. Ridge on 61 mostly-irrelevant columns had to
over-shrink; ARD pruned them instead and recovered FR rho 0.2666. Give ridge six columns
that actually carry French signal and it reaches FR rho **0.2874** on its own -- better
than ARD ever managed -- and the relevance-determination advantage has nothing left to do.
Section 7's advice ("anyone working on France should start from ARD") is **withdrawn**:
start from the cumulative-return features instead, and then ridge is fine.

**The capacity finding does survive, which is the one that mattered.** LightGBM keeps
+0.0187 at 15/15 on the richer feature set, and its DE rho is still the highest of any
model tried (0.7838 against ridge's 0.7595). "Added capacity always loses" stays dead.

**The blend survives but shrinks.** `0.5 LGBM + 0.5 ARD` is still the best combination at
+0.0233 (15/15), but its margin over LightGBM alone falls from +0.0130 to +0.0046, exactly
as you would expect once ARD stops being a distinct source of France signal.

Read sections 1-9 as measured against the 0.5231 feature set. Where this section and those
disagree, this one wins. It is also the third feature regime this project has been through
in a day, so the honest summary of the *method* is: the boosting result has now survived two
regime changes, the ARD result survived one and died in the next, and nothing here should be
assumed to transfer to a fourth without being re-run -- it takes about three minutes.
