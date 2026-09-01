# Model zoo: what beats, ties, or usefully diversifies from the ridge baseline

~50 model configurations across 13 families, all fitted per country, all on the
rank-transformed target, all scored on the same fold assignments as the ridge baseline
(**0.2909 +/- 0.0069**, reproduced exactly). Headline claims are re-confirmed on 30
fold seeds that were never used for selection.

## TL;DR

1. **No single model family beats ridge outright.** The best honest single models --
   `Ridge-winsor` (+0.0024), `Huber` (+0.0020), `PLS` (+0.0014) -- are inside the
   +/-0.007 noise band, and they win by being *more* constrained, not less. Capacity
   keeps losing: SVR-rbf -0.014, boosting -0.009 to -0.027, random forest -0.014,
   MLP -0.076, GP with a fitted RBF kernel -0.058.
2. **Diversity beats accuracy, and it is not close.** The models that help most in a blend
   are the ones that score *worst* alone. `0.6*z(ridge) + 0.4*z(LightGBM)` scores 0.2984
   (+0.0075, 15/15) even though that LightGBM is 0.2784 by itself; `0.75 ridge + 0.25
   HistGB` gains +0.0062 (30/30 fresh seeds) even though HistGB alone is 0.2631 -- 0.027
   *below* ridge. The best configuration found is
   **`0.50*z(ridge) + 0.25*z(LightGBM) + 0.25*z(SVR-rbf)`: 0.3009 on seeds 0..14
   (+0.0100, 15/15), confirmed at 0.2981 on 30 unseen seeds (+0.0089, sd 0.0031,
   30/30)**. That is the only result in this sweep that clears the +/-0.007 noise band
   outright, and both of its non-ridge members lose to ridge by 0.013 on their own.
3. **"Capacity always loses" is a France rule, not a Germany rule.** Split by country,
   several families beat ridge *on Germany alone*: `Spline+Ridge` +0.0082 DE (14/15),
   `KRR-rbf` +0.0080 DE (13/15), `ARDRegression` +0.0077 DE (13/15). All three lose on
   France. Ridge for FR + a mildly nonlinear model for DE scores 0.2968 (+0.0059, 14/15)
   and **replicates on 30 fresh seeds at +0.0035 to +0.0038**. The amount of nonlinearity
   Germany tolerates is tiny: inner CV picks `n_knots=3` for the DE splines in 20/20 folds
   and `gamma=1e-3` for DE kernel ridge in 17/20, i.e. a kernel that is 94% linear.
4. **Do not rank-normalise the two countries' predictions separately before pooling.**
   It costs 0.014 (0.2909 -> 0.2768). Both heads are fitted on a within-country uniform
   rank target, so their relative *level* is real signal. Blends here standardise each
   model's OOF vector globally instead.

## Protocol

* **Baseline.** Per-country `RidgeCV(alphas=np.logspace(-2,4,40))` on median-imputed raw
  features, rank-transformed target, FR = `[FR_WINDPOW, GAS_RET, CARBON_RET]`, DE = all 29.
  Reproduced exactly: **0.2909 +/- 0.0069** over 15 randomised grouped-CV seeds.
* **Pairing.** Every candidate is run on the *same* `P.make_folds(train.DAY_ID, seed=s)`
  assignments, seeds 0..14 (10 for the ARD GPs). Reported `delta` is the mean of the 15
  per-seed differences, `sd(delta)` their standard deviation, `wins/N` the count of seeds
  where the candidate is higher.
* **Noise band.** The absolute score moves +/-0.007 across seeds, so no absolute score
  difference below that is interpretable on its own. Paired deltas are far tighter (a
  matched sd of 0.002-0.005 is typical), so `delta` + `wins/N` is the column to read.
  Both are quoted.
* **Leakage.** Imputation medians, standardisation, winsorisation cut-points, feature-rank
  grids, spline knots and *all* hyperparameter selection are fitted on the fold's training
  rows of that country only. Inner tuning is a hand-rolled `Tuned` wrapper: 3-fold
  `KFold` inside the fold, scored by Spearman. Plain `KFold` is safe inside a country
  because each country has exactly one row per `DAY_ID` (FR 851 rows / 851 days,
  DE 643 / 643); the day grouping only matters for the outer split, which uses it.
* **One deliberate exception.** Trees and boosting are scored as *fixed* heavily
  regularised configurations rather than inner-tuned, and the best of several is reported.
  That is optimistic by construction, and it is the right protocol for a family expected
  to lose: an upper bound that still loses is conclusive. None of them won, so none needed
  a re-run under honest tuning.
* **Selection inflation is handled.** The per-country and blend results below were found by
  scanning ~50 models on seeds 0..14, so they are selection-inflated. Every headline claim
  was re-scored on **30 fresh fold seeds (15..44)** with the configuration fixed in
  advance; those numbers are in the confirmation section and are the ones to trust.

## 1. Full sweep, paired against ridge (seeds 0..14)

`delta` is the mean per-seed difference from ridge on identical folds.
`corr(ridge)` is the pooled Spearman correlation between the model's OOF
predictions and ridge's, averaged over seeds.

| model | family | config | score | delta vs ridge | sd(delta) | wins/N | corr(ridge) |
|---|---|---|---|---|---|---|---|
| `Ridge-winsor` | linear (robust) | features clipped to fold-train 2/98 pct, RidgeCV | 0.2933 | +0.0024 | 0.0019 | 13/15 | 0.994 |
| `Huber` | linear (robust) | alpha tuned inner, epsilon=1.35 | 0.2929 | +0.0020 | 0.0038 | 10/15 | 0.983 |
| `PLS` | projection | n_components tuned inner | 0.2923 | +0.0014 | 0.0068 | 9/15 | 0.976 |
| `ARDRegression` | linear | default | 0.2918 | +0.0009 | 0.0037 | 10/15 | 0.965 |
| `GP-Matern32` | GP | C*Matern(nu=1.5,iso)+White | 0.2913 | +0.0004 | 0.0051 | 6/15 | 0.942 |
| `BayesianRidge` | linear | default ARD-free priors | 0.2904 | -0.0005 | 0.0019 | 5/15 | 0.995 |
| `GP-linear` | GP | C*DotProduct+White (Bayesian linear GP) | 0.2904 | -0.0005 | 0.0019 | 5/15 | 0.995 |
| `Spline+Ridge` | additive | cubic B-splines per feature, n_knots tuned inner, RidgeCV head | 0.2900 | -0.0009 | 0.0057 | 8/15 | 0.934 |
| `RFF+Ridge` | kernel | random Fourier features, gamma/D tuned inner | 0.2886 | -0.0023 | 0.0049 | 4/15 | 0.959 |
| `KRR-rbf` | kernel | alpha/gamma tuned inner | 0.2875 | -0.0034 | 0.0107 | 5/15 | 0.930 |
| `Nystroem+Ridge` | kernel | Nystroem rbf, gamma/n_comp tuned inner | 0.2873 | -0.0036 | 0.0053 | 4/15 | 0.960 |
| `GP-RQ` | GP | C*RationalQuadratic+White | 0.2866 | -0.0043 | 0.0057 | 3/15 | 0.943 |
| `LassoLarsCV` | linear | LARS path, 5-fold inner | 0.2860 | -0.0049 | 0.0045 | 1/15 | 0.962 |
| `SVR-linear` | SVM | C/eps tuned inner | 0.2860 | -0.0049 | 0.0049 | 2/15 | 0.971 |
| `LassoCV` | linear | 100 alphas, 5-fold inner | 0.2858 | -0.0051 | 0.0045 | 1/15 | 0.962 |
| `ElasticNetCV` | linear | l1_ratio grid x 60 alphas, 5-fold inner | 0.2855 | -0.0053 | 0.0041 | 1/15 | 0.964 |
| `ET-d4-leaf20` | trees | 400 extra trees, max_depth 4, min_leaf 20, max_feat .4/1.0 | 0.2854 | -0.0055 | 0.0050 | 0/15 | 0.907 |
| `ET-leaf50` | trees | 400 extra trees, unbounded depth, min_leaf 50, max_feat 1.0 | 0.2850 | -0.0059 | 0.0066 | 1/15 | 0.891 |
| `GP-lin+RBF` | GP | C*Dot + C*RBF(iso) + White | 0.2849 | -0.0060 | 0.0048 | 2/15 | 0.956 |
| `LGBM-stumps` | trees | num_leaves 2, lr .02, 500 rounds, l2=20, ff .4/1.0, bagging .7 | 0.2819 | -0.0090 | 0.0063 | 0/15 | 0.870 |
| `Ridge-rankfeat` | linear (robust) | RidgeCV on within-fold rank-transformed features | 0.2807 | -0.0102 | 0.0043 | 0/15 | 0.962 |
| `XGB-depth1` | trees | depth 1, lr .02, 500 rounds, mcw 10, l2=20, colsample .4/1.0 | 0.2796 | -0.0113 | 0.0053 | 0/15 | 0.871 |
| `RF-leaf50` | trees | 400 trees, unbounded depth, min_leaf 50, max_feat .3/1.0 | 0.2786 | -0.0123 | 0.0065 | 0/15 | 0.858 |
| `LGBM-leaves4` | trees | num_leaves 4, lr .02, 300 rounds, l2=20, ff .4/1.0 | 0.2784 | -0.0125 | 0.0074 | 0/15 | 0.825 |
| `CatBoost-d3` | trees | depth 3, lr .03, 400 iters, l2_leaf_reg 20 | 0.2784 | -0.0125 | 0.0057 | 0/15 | 0.838 |
| `XGB-depth2` | trees | depth 2, lr .02, 300 rounds, mcw 20, l2=20, colsample .4/1.0 | 0.2783 | -0.0126 | 0.0064 | 1/15 | 0.834 |
| `GP-RBF-ard` | GP | C*RBF(ARD)+White, L-BFGS capped at 50 iters, 10 seeds | 0.2779 | -0.0107 | 0.0091 | 1/10 | 0.819 |
| `SGD-eps-insensitive` | linear (robust) | SGD, epsilon-insensitive loss (LAD-like), alpha tuned inner | 0.2779 | -0.0130 | 0.0096 | 1/15 | 0.941 |
| `Quantile-median` | linear (robust) | q=0.5 LAD, alpha tuned inner | 0.2776 | -0.0133 | 0.0081 | 1/15 | 0.945 |
| `Spline+Ridge-rankfeat` | additive | splines on rank features, n_knots tuned inner | 0.2776 | -0.0133 | 0.0078 | 0/15 | 0.881 |
| `SVR-rbf` | SVM | C/gamma tuned inner, eps=0.1 | 0.2772 | -0.0137 | 0.0062 | 0/15 | 0.850 |
| `RF-d4-leaf20` | trees | 400 trees, max_depth 4, min_leaf 20, max_feat .4 (DE) / 1.0 (FR) | 0.2771 | -0.0138 | 0.0062 | 0/15 | 0.839 |
| `NuSVR-rbf` | SVM | C/gamma/nu tuned inner | 0.2766 | -0.0143 | 0.0110 | 0/15 | 0.839 |
| `PCA+Ridge` | projection | n_comp tuned inner, RidgeCV head | 0.2765 | -0.0144 | 0.0055 | 0/15 | 0.960 |
| `KRR-laplacian` | kernel | alpha/gamma tuned inner | 0.2747 | -0.0162 | 0.0094 | 1/15 | 0.850 |
| `OMP-CV` | linear | orthogonal matching pursuit, n_nonzero by inner CV | 0.2740 | -0.0169 | 0.0102 | 1/15 | 0.916 |
| `RANSAC-Ridge` | linear (robust) | RANSAC over RidgeCV base, 100 trials | 0.2738 | -0.0171 | 0.0061 | 0/15 | 0.945 |
| `GP-RBF-ard-rankfeat` | GP | rank feats, C*RBF(ARD)+White, capped, 10 seeds | 0.2738 | -0.0148 | 0.0102 | 0/10 | 0.882 |
| `KRR-rbf-rankfeat` | kernel | kernel ridge rbf on rank feats, alpha/gamma tuned inner | 0.2732 | -0.0177 | 0.0088 | 0/15 | 0.905 |
| `TheilSen` | linear (robust) | median-of-slopes, max_subpopulation 2000 | 0.2732 | -0.0177 | 0.0093 | 0/15 | 0.914 |
| `CatBoost-d1` | trees | depth 1, lr .03, 500 iters, l2_leaf_reg 20 | 0.2713 | -0.0196 | 0.0058 | 0/15 | 0.865 |
| `GBR-stumps` | trees | depth-1 boosting, lr .02, 400 rounds, subsample .7 | 0.2689 | -0.0220 | 0.0064 | 0/15 | 0.849 |
| `SGD-huber` | linear (robust) | SGD, huber loss, l2, alpha tuned inner | 0.2682 | -0.0227 | 0.0076 | 0/15 | 0.928 |
| `HGB-shallow` | trees | hist GB, depth 2, lr .03, 300 iters, l2=10, min_leaf 30 | 0.2635 | -0.0274 | 0.0091 | 0/15 | 0.776 |
| `kNN-std-dist` | kNN | z-scored feats, distance weights, k tuned | 0.2530 | -0.0379 | 0.0102 | 0/15 | 0.810 |
| `SVR-rbf-rankfeat` | SVM | rank feats, C/gamma tuned inner | 0.2497 | -0.0412 | 0.0100 | 0/15 | 0.741 |
| `kNN-rank-L1` | kNN | rank feats, manhattan, distance weights, k tuned | 0.2447 | -0.0462 | 0.0140 | 0/15 | 0.805 |
| `kNN-PCA-rank` | kNN | PCA on rank feats then kNN, n_comp/k tuned inner | 0.2418 | -0.0491 | 0.0101 | 0/15 | 0.786 |
| `kNN-rank-dist` | kNN | rank feats, distance weights, k tuned inner | 0.2417 | -0.0492 | 0.0117 | 0/15 | 0.807 |
| `MLP-small` | neural net | 1 hidden layer, heavy L2, early stopping, size/alpha tuned inner | 0.2349 | -0.0559 | 0.0141 | 0/15 | 0.777 |
| `GP-RBF-iso` | GP | C*RBF(iso)+White, MLL-fitted in fold | 0.2327 | -0.0582 | 0.0241 | 0/15 | 0.832 |
| `kNN-rank-unif` | kNN | rank feats, uniform weights, k tuned inner | 0.2265 | -0.0644 | 0.0118 | 0/15 | 0.749 |
| `KRR-poly2` | kernel | degree 2, alpha/gamma tuned inner | 0.2238 | -0.0671 | 0.0291 | 0/15 | 0.704 |
| `MLP-tiny-fixed` | neural net | (4,), alpha=10, early stopping, no tuning | 0.2145 | -0.0764 | 0.0185 | 0/15 | 0.725 |


### 1b. Flagged for diversity: corr(ridge) < 0.90 and score > 0.24

These are the models worth keeping as ensemble members even though every one of
them loses to ridge on its own.

| model | score | corr(ridge) | best blend gain with ridge |
|---|---|---|---|
| `SVR-rbf-rankfeat` | 0.2497 | 0.741 | +0.0051 at w=0.25 |
| `HGB-shallow` | 0.2635 | 0.776 | +0.0060 at w=0.25 |
| `kNN-PCA-rank` | 0.2418 | 0.786 | -0.0001 at w=0.25 |
| `kNN-rank-L1` | 0.2447 | 0.805 | -0.0005 at w=0.25 |
| `kNN-rank-dist` | 0.2417 | 0.807 | -0.0012 at w=0.25 |
| `kNN-std-dist` | 0.2530 | 0.810 | +0.0042 at w=0.25 |
| `GP-RBF-ard` | 0.2779 | 0.819 | +0.0089 at w=0.4 |
| `LGBM-leaves4` | 0.2784 | 0.825 | +0.0075 at w=0.4 |
| `XGB-depth2` | 0.2783 | 0.834 | +0.0060 at w=0.4 |
| `CatBoost-d3` | 0.2784 | 0.838 | +0.0052 at w=0.4 |
| `RF-d4-leaf20` | 0.2771 | 0.839 | +0.0043 at w=0.25 |
| `NuSVR-rbf` | 0.2766 | 0.839 | +0.0063 at w=0.4 |
| `GBR-stumps` | 0.2689 | 0.849 | +0.0018 at w=0.25 |
| `KRR-laplacian` | 0.2747 | 0.850 | +0.0036 at w=0.25 |
| `SVR-rbf` | 0.2772 | 0.850 | +0.0064 at w=0.4 |
| `RF-leaf50` | 0.2786 | 0.858 | +0.0036 at w=0.25 |
| `CatBoost-d1` | 0.2713 | 0.865 | +0.0008 at w=0.25 |
| `LGBM-stumps` | 0.2819 | 0.870 | +0.0045 at w=0.4 |
| `XGB-depth1` | 0.2796 | 0.871 | +0.0035 at w=0.25 |
| `Spline+Ridge-rankfeat` | 0.2776 | 0.881 | +0.0030 at w=0.25 |
| `GP-RBF-ard-rankfeat` | 0.2738 | 0.882 | +0.0026 at w=0.25 |
| `ET-leaf50` | 0.2850 | 0.891 | +0.0026 at w=0.4 |


## 2. Per-country breakdown

Ridge scores FR rho 0.2084 and DE rho 0.3598. Sorted by DE.

| model | FR rho | delta | DE rho | delta | corr(ridge) FR | corr(ridge) DE |
|---|---|---|---|---|---|---|
| `Spline+Ridge` | 0.1923 | -0.0161 (0/15) | 0.3680 | +0.0082 (14/15) | 0.947 | 0.928 |
| `KRR-rbf` | 0.1907 | -0.0177 (3/15) | 0.3677 | +0.0080 (13/15) | 0.853 | 0.989 |
| `ARDRegression` | 0.2023 | -0.0061 (0/15) | 0.3675 | +0.0077 (13/15) | 0.989 | 0.954 |
| `Huber` | 0.2109 | +0.0025 (15/15) | 0.3631 | +0.0033 (9/15) | 0.995 | 0.981 |
| `Ridge-winsor` | 0.2097 | +0.0012 (13/15) | 0.3630 | +0.0032 (13/15) | 0.995 | 0.995 |
| `PLS` | 0.2114 | +0.0030 (12/15) | 0.3615 | +0.0018 (10/15) | 0.989 | 0.969 |
| `Nystroem+Ridge` | 0.2007 | -0.0077 (5/15) | 0.3606 | +0.0008 (9/15) | 0.925 | 0.988 |
| `RFF+Ridge` | 0.2044 | -0.0040 (4/15) | 0.3600 | +0.0002 (7/15) | 0.934 | 0.980 |
| `BayesianRidge` | 0.2085 | +0.0000 (8/15) | 0.3588 | -0.0010 (6/15) | 1.000 | 0.994 |
| `GP-linear` | 0.2085 | +0.0000 (9/15) | 0.3588 | -0.0010 (6/15) | 1.000 | 0.994 |
| `GP-RQ` | 0.2035 | -0.0050 (6/15) | 0.3557 | -0.0041 (2/15) | 0.897 | 0.982 |
| `GP-Matern32` | 0.2149 | +0.0065 (11/15) | 0.3552 | -0.0046 (1/15) | 0.904 | 0.977 |
| `KRR-poly2` | 0.0927 | -0.1158 (0/15) | 0.3547 | -0.0051 (8/15) | 0.522 | 0.933 |
| `LassoLarsCV` | 0.2072 | -0.0012 (3/15) | 0.3544 | -0.0054 (2/15) | 0.999 | 0.946 |
| `LassoCV` | 0.2067 | -0.0017 (2/15) | 0.3543 | -0.0055 (2/15) | 0.998 | 0.946 |
| `GP-lin+RBF` | 0.2042 | -0.0043 (5/15) | 0.3539 | -0.0059 (1/15) | 0.910 | 0.990 |
| `ElasticNetCV` | 0.2070 | -0.0015 (2/15) | 0.3531 | -0.0066 (2/15) | 0.999 | 0.949 |
| `XGB-depth2` | 0.1920 | -0.0165 (0/15) | 0.3529 | -0.0069 (5/15) | 0.817 | 0.849 |
| `SVR-linear` | 0.2121 | +0.0037 (13/15) | 0.3523 | -0.0074 (3/15) | 0.988 | 0.968 |
| `LGBM-stumps` | 0.1950 | -0.0134 (1/15) | 0.3520 | -0.0077 (3/15) | 0.879 | 0.865 |
| `LGBM-leaves4` | 0.1926 | -0.0158 (1/15) | 0.3507 | -0.0090 (4/15) | 0.796 | 0.846 |
| `XGB-depth1` | 0.1934 | -0.0150 (0/15) | 0.3499 | -0.0099 (1/15) | 0.883 | 0.864 |
| `Quantile-median` | 0.2126 | +0.0041 (14/15) | 0.3498 | -0.0100 (3/15) | 0.991 | 0.939 |
| `RF-d4-leaf20` | 0.1905 | -0.0179 (0/15) | 0.3493 | -0.0105 (3/15) | 0.834 | 0.859 |
| `KRR-laplacian` | 0.1903 | -0.0182 (2/15) | 0.3488 | -0.0110 (1/15) | 0.809 | 0.889 |
| `SGD-eps-insensitive` | 0.2099 | +0.0015 (9/15) | 0.3477 | -0.0121 (3/15) | 0.982 | 0.935 |
| `SVR-rbf` | 0.1979 | -0.0105 (4/15) | 0.3467 | -0.0131 (1/15) | 0.839 | 0.872 |
| `RF-leaf50` | 0.1994 | -0.0090 (3/15) | 0.3461 | -0.0136 (1/15) | 0.866 | 0.876 |
| `SGD-huber` | 0.1995 | -0.0089 (3/15) | 0.3460 | -0.0138 (0/15) | 0.943 | 0.952 |
| `GP-RBF-ard` | 0.2010 | -0.0076 (2/10) | 0.3457 | -0.0095 (4/10) | 0.902 | 0.767 |
| `Spline+Ridge-rankfeat` | 0.1961 | -0.0123 (1/15) | 0.3437 | -0.0161 (1/15) | 0.927 | 0.848 |
| `GP-RBF-ard-rankfeat` | 0.1939 | -0.0147 (1/10) | 0.3433 | -0.0120 (2/10) | 0.924 | 0.854 |
| `NuSVR-rbf` | 0.1983 | -0.0101 (4/15) | 0.3426 | -0.0172 (1/15) | 0.814 | 0.864 |
| `ET-leaf50` | 0.2033 | -0.0051 (2/15) | 0.3425 | -0.0172 (1/15) | 0.959 | 0.886 |
| `OMP-CV` | 0.1978 | -0.0106 (0/15) | 0.3405 | -0.0193 (1/15) | 0.982 | 0.883 |
| `PCA+Ridge` | 0.2034 | -0.0050 (4/15) | 0.3387 | -0.0210 (0/15) | 0.968 | 0.957 |
| `KRR-rbf-rankfeat` | 0.1919 | -0.0166 (2/15) | 0.3387 | -0.0211 (0/15) | 0.884 | 0.921 |
| `ET-d4-leaf20` | 0.2096 | +0.0012 (9/15) | 0.3385 | -0.0213 (0/15) | 0.954 | 0.899 |
| `Ridge-rankfeat` | 0.2099 | +0.0015 (11/15) | 0.3380 | -0.0218 (0/15) | 0.972 | 0.959 |
| `HGB-shallow` | 0.1777 | -0.0307 (0/15) | 0.3375 | -0.0222 (0/15) | 0.759 | 0.781 |
| `TheilSen` | 0.2037 | -0.0047 (3/15) | 0.3371 | -0.0227 (0/15) | 0.974 | 0.893 |
| `RANSAC-Ridge` | 0.2085 | +0.0001 (8/15) | 0.3355 | -0.0243 (0/15) | 0.973 | 0.938 |
| `CatBoost-d3` | 0.2137 | +0.0053 (9/15) | 0.3331 | -0.0267 (0/15) | 0.843 | 0.836 |
| `CatBoost-d1` | 0.1979 | -0.0105 (0/15) | 0.3290 | -0.0308 (0/15) | 0.906 | 0.840 |
| `GBR-stumps` | 0.2004 | -0.0081 (3/15) | 0.3223 | -0.0375 (0/15) | 0.873 | 0.836 |
| `MLP-small` | 0.1914 | -0.0171 (2/15) | 0.3034 | -0.0564 (0/15) | 0.852 | 0.782 |
| `SVR-rbf-rankfeat` | 0.1910 | -0.0174 (2/15) | 0.3026 | -0.0572 (0/15) | 0.789 | 0.718 |
| `kNN-std-dist` | 0.2167 | +0.0083 (12/15) | 0.3018 | -0.0580 (0/15) | 0.864 | 0.854 |
| `kNN-rank-L1` | 0.2048 | -0.0036 (10/15) | 0.2885 | -0.0713 (0/15) | 0.845 | 0.836 |
| `kNN-rank-dist` | 0.2050 | -0.0034 (10/15) | 0.2816 | -0.0782 (0/15) | 0.853 | 0.824 |
| `MLP-tiny-fixed` | 0.1792 | -0.0292 (2/15) | 0.2765 | -0.0833 (0/15) | 0.826 | 0.722 |
| `kNN-rank-unif` | 0.1938 | -0.0146 (3/15) | 0.2696 | -0.0902 (0/15) | 0.777 | 0.790 |
| `GP-RBF-iso` | 0.2047 | -0.0038 (5/15) | 0.2602 | -0.0996 (0/15) | 0.929 | 0.756 |
| `kNN-PCA-rank` | 0.2107 | +0.0023 (10/15) | 0.2585 | -0.1013 (0/15) | 0.883 | 0.709 |


## 3. Blend with ridge (pooled-z, seeds 0..14)

`(1-w)*z(ridge) + w*z(candidate)`, scored pooled. This is where diversity pays.
Weights are scanned here, so these are selection-inflated; see section 5.

| candidate | corr(ridge) | blend w=0.25 | delta | blend w=0.40 | delta | blend w=0.50 | delta |
|---|---|---|---|---|---|---|---|
| `LGBM-leaves4` | 0.825 | 0.2978 | +0.0069 (15/15) | 0.2984 | +0.0075 (15/15) | 0.2976 | +0.0067 (14/15) |
| `GP-RBF-ard` | 0.819 | 0.2960 | +0.0074 (10/10) | 0.2975 | +0.0089 (10/10) | 0.2973 | +0.0087 (9/10) |
| `SVR-rbf` | 0.850 | 0.2967 | +0.0058 (15/15) | 0.2973 | +0.0064 (15/15) | 0.2964 | +0.0055 (13/15) |
| `NuSVR-rbf` | 0.839 | 0.2967 | +0.0058 (15/15) | 0.2972 | +0.0063 (13/15) | 0.2963 | +0.0054 (12/15) |
| `HGB-shallow` | 0.776 | 0.2969 | +0.0060 (15/15) | 0.2962 | +0.0053 (14/15) | 0.2938 | +0.0029 (12/15) |
| `XGB-depth2` | 0.834 | 0.2966 | +0.0057 (15/15) | 0.2969 | +0.0060 (15/15) | 0.2960 | +0.0051 (14/15) |
| `CatBoost-d3` | 0.838 | 0.2957 | +0.0049 (14/15) | 0.2961 | +0.0052 (14/15) | 0.2951 | +0.0042 (13/15) |
| `SVR-rbf-rankfeat` | 0.741 | 0.2959 | +0.0051 (15/15) | 0.2937 | +0.0028 (12/15) | 0.2898 | -0.0011 (7/15) |
| `GP-Matern32` | 0.942 | 0.2944 | +0.0035 (15/15) | 0.2954 | +0.0045 (15/15) | 0.2956 | +0.0047 (15/15) |
| `LGBM-stumps` | 0.870 | 0.2952 | +0.0043 (15/15) | 0.2954 | +0.0045 (13/15) | 0.2946 | +0.0038 (12/15) |
| `Spline+Ridge` | 0.934 | 0.2945 | +0.0036 (14/15) | 0.2953 | +0.0044 (14/15) | 0.2953 | +0.0044 (14/15) |
| `RF-d4-leaf20` | 0.839 | 0.2952 | +0.0043 (15/15) | 0.2950 | +0.0041 (13/15) | 0.2941 | +0.0033 (13/15) |
| `kNN-std-dist` | 0.810 | 0.2951 | +0.0042 (14/15) | 0.2929 | +0.0020 (11/15) | 0.2895 | -0.0014 (4/15) |
| `RF-leaf50` | 0.858 | 0.2945 | +0.0036 (14/15) | 0.2942 | +0.0033 (14/15) | 0.2933 | +0.0024 (12/15) |
| `KRR-laplacian` | 0.850 | 0.2945 | +0.0036 (14/15) | 0.2942 | +0.0033 (12/15) | 0.2930 | +0.0021 (8/15) |
| `XGB-depth1` | 0.871 | 0.2944 | +0.0035 (14/15) | 0.2943 | +0.0034 (14/15) | 0.2934 | +0.0025 (12/15) |
| `KRR-rbf` | 0.930 | 0.2935 | +0.0026 (13/15) | 0.2941 | +0.0032 (13/15) | 0.2941 | +0.0032 (11/15) |
| `ET-d4-leaf20` | 0.907 | 0.2935 | +0.0026 (14/15) | 0.2939 | +0.0031 (13/15) | 0.2938 | +0.0029 (12/15) |
| `Spline+Ridge-rankfeat` | 0.881 | 0.2939 | +0.0030 (13/15) | 0.2935 | +0.0026 (11/15) | 0.2925 | +0.0016 (10/15) |
| `ET-leaf50` | 0.891 | 0.2932 | +0.0023 (13/15) | 0.2934 | +0.0026 (13/15) | 0.2934 | +0.0025 (12/15) |
| `PLS` | 0.976 | 0.2926 | +0.0017 (11/15) | 0.2931 | +0.0022 (11/15) | 0.2934 | +0.0025 (10/15) |
| `ARDRegression` | 0.965 | 0.2926 | +0.0017 (14/15) | 0.2932 | +0.0023 (14/15) | 0.2934 | +0.0025 (13/15) |
| `Huber` | 0.983 | 0.2925 | +0.0016 (15/15) | 0.2931 | +0.0022 (13/15) | 0.2933 | +0.0025 (13/15) |
| `GP-RQ` | 0.943 | 0.2929 | +0.0020 (14/15) | 0.2932 | +0.0023 (12/15) | 0.2928 | +0.0019 (11/15) |
| `GBR-stumps` | 0.849 | 0.2927 | +0.0018 (13/15) | 0.2917 | +0.0008 (10/15) | 0.2903 | -0.0006 (6/15) |
| `RFF+Ridge` | 0.959 | 0.2924 | +0.0015 (12/15) | 0.2926 | +0.0018 (11/15) | 0.2926 | +0.0017 (11/15) |
| `Ridge-winsor` | 0.994 | 0.2917 | +0.0008 (14/15) | 0.2921 | +0.0013 (14/15) | 0.2924 | +0.0015 (14/15) |
| `Nystroem+Ridge` | 0.960 | 0.2919 | +0.0010 (10/15) | 0.2919 | +0.0010 (10/15) | 0.2916 | +0.0007 (8/15) |
| `CatBoost-d1` | 0.865 | 0.2917 | +0.0008 (11/15) | 0.2905 | -0.0004 (9/15) | 0.2891 | -0.0018 (5/15) |
| `TheilSen` | 0.914 | 0.2915 | +0.0006 (8/15) | 0.2903 | -0.0006 (6/15) | 0.2888 | -0.0021 (5/15) |
| `GP-lin+RBF` | 0.956 | 0.2914 | +0.0005 (10/15) | 0.2910 | +0.0001 (8/15) | 0.2904 | -0.0004 (6/15) |
| `SVR-linear` | 0.971 | 0.2914 | +0.0005 (11/15) | 0.2911 | +0.0002 (10/15) | 0.2907 | -0.0001 (6/15) |
| `LassoLarsCV` | 0.962 | 0.2914 | +0.0005 (11/15) | 0.2911 | +0.0003 (9/15) | 0.2907 | -0.0002 (7/15) |
| `LassoCV` | 0.962 | 0.2913 | +0.0004 (11/15) | 0.2910 | +0.0001 (8/15) | 0.2906 | -0.0003 (7/15) |
| `GP-RBF-ard-rankfeat` | 0.882 | 0.2912 | +0.0026 (8/10) | 0.2907 | +0.0021 (7/10) | 0.2894 | +0.0008 (5/10) |
| `ElasticNetCV` | 0.964 | 0.2911 | +0.0002 (7/15) | 0.2908 | -0.0001 (6/15) | 0.2903 | -0.0006 (6/15) |
| `BayesianRidge` | 0.995 | 0.2910 | +0.0001 (8/15) | 0.2911 | +0.0002 (8/15) | 0.2910 | +0.0001 (7/15) |
| `GP-linear` | 0.995 | 0.2910 | +0.0001 (8/15) | 0.2911 | +0.0002 (8/15) | 0.2910 | +0.0001 (7/15) |
| `KRR-rbf-rankfeat` | 0.905 | 0.2910 | +0.0001 (7/15) | 0.2895 | -0.0014 (6/15) | 0.2879 | -0.0030 (4/15) |
| `Quantile-median` | 0.945 | 0.2910 | +0.0001 (7/15) | 0.2902 | -0.0007 (4/15) | 0.2892 | -0.0017 (3/15) |
| `SGD-eps-insensitive` | 0.941 | 0.2909 | +0.0000 (9/15) | 0.2898 | -0.0011 (7/15) | 0.2886 | -0.0023 (6/15) |
| `kNN-PCA-rank` | 0.786 | 0.2908 | -0.0001 (10/15) | 0.2868 | -0.0040 (3/15) | 0.2828 | -0.0081 (1/15) |
| `kNN-rank-L1` | 0.805 | 0.2904 | -0.0005 (8/15) | 0.2863 | -0.0046 (2/15) | 0.2818 | -0.0091 (1/15) |
| `GP-RBF-iso` | 0.832 | 0.2903 | -0.0005 (9/15) | 0.2873 | -0.0036 (2/15) | 0.2839 | -0.0070 (1/15) |
| `OMP-CV` | 0.916 | 0.2901 | -0.0008 (5/15) | 0.2887 | -0.0022 (3/15) | 0.2873 | -0.0036 (3/15) |
| `kNN-rank-dist` | 0.807 | 0.2897 | -0.0012 (7/15) | 0.2851 | -0.0057 (0/15) | 0.2805 | -0.0104 (0/15) |
| `RANSAC-Ridge` | 0.945 | 0.2897 | -0.0012 (2/15) | 0.2880 | -0.0029 (2/15) | 0.2864 | -0.0045 (2/15) |
| `Ridge-rankfeat` | 0.962 | 0.2896 | -0.0013 (2/15) | 0.2885 | -0.0024 (1/15) | 0.2874 | -0.0034 (0/15) |
| `PCA+Ridge` | 0.960 | 0.2893 | -0.0015 (1/15) | 0.2877 | -0.0032 (1/15) | 0.2864 | -0.0045 (1/15) |
| `SGD-huber` | 0.928 | 0.2891 | -0.0018 (3/15) | 0.2867 | -0.0041 (1/15) | 0.2847 | -0.0062 (1/15) |
| `kNN-rank-unif` | 0.749 | 0.2887 | -0.0022 (3/15) | 0.2828 | -0.0081 (2/15) | 0.2770 | -0.0139 (1/15) |
| `MLP-small` | 0.777 | 0.2886 | -0.0022 (3/15) | 0.2836 | -0.0073 (1/15) | 0.2785 | -0.0124 (1/15) |
| `KRR-poly2` | 0.704 | 0.2862 | -0.0047 (3/15) | 0.2757 | -0.0152 (1/15) | 0.2665 | -0.0244 (0/15) |
| `MLP-tiny-fixed` | 0.725 | 0.2862 | -0.0047 (1/15) | 0.2791 | -0.0118 (1/15) | 0.2725 | -0.0184 (0/15) |


## 4. FR/DE splice (seeds 0..14)

Raw predictions, ridge for one country and the candidate for the other.

| model | ridgeFR + candDE | delta | candFR + ridgeDE | delta |
|---|---|---|---|---|
| `Spline+Ridge` | 0.2968 | +0.0059 (14/15) | 0.2841 | -0.0068 (0/15) |
| `KRR-rbf` | 0.2963 | +0.0054 (13/15) | 0.2819 | -0.0090 (3/15) |
| `ARDRegression` | 0.2941 | +0.0032 (12/15) | 0.2886 | -0.0023 (0/15) |
| `Ridge-winsor` | 0.2926 | +0.0017 (12/15) | 0.2914 | +0.0005 (10/15) |
| `Nystroem+Ridge` | 0.2924 | +0.0015 (10/15) | 0.2861 | -0.0048 (3/15) |
| `Huber` | 0.2920 | +0.0011 (9/15) | 0.2906 | -0.0002 (6/15) |
| `RFF+Ridge` | 0.2916 | +0.0007 (8/15) | 0.2881 | -0.0028 (3/15) |
| `PLS` | 0.2913 | +0.0004 (9/15) | 0.2915 | +0.0006 (11/15) |
| `GP-RQ` | 0.2906 | -0.0003 (6/15) | 0.2878 | -0.0031 (3/15) |
| `BayesianRidge` | 0.2904 | -0.0005 (6/15) | 0.2909 | +0.0000 (8/15) |
| `GP-linear` | 0.2904 | -0.0005 (6/15) | 0.2909 | +0.0000 (8/15) |
| `GP-Matern32` | 0.2904 | -0.0005 (5/15) | 0.2927 | +0.0018 (11/15) |
| `LGBM-stumps` | 0.2892 | -0.0017 (7/15) | 0.2846 | -0.0063 (0/15) |
| `KRR-poly2` | 0.2884 | -0.0025 (11/15) | 0.2263 | -0.0646 (0/15) |
| `GP-lin+RBF` | 0.2883 | -0.0026 (2/15) | 0.2879 | -0.0030 (4/15) |
| `RF-d4-leaf20` | 0.2883 | -0.0026 (7/15) | 0.2823 | -0.0086 (0/15) |
| `XGB-depth2` | 0.2880 | -0.0029 (5/15) | 0.2819 | -0.0090 (0/15) |
| `LGBM-leaves4` | 0.2877 | -0.0032 (4/15) | 0.2820 | -0.0089 (1/15) |
| `XGB-depth1` | 0.2876 | -0.0033 (2/15) | 0.2836 | -0.0073 (0/15) |
| `LassoLarsCV` | 0.2869 | -0.0040 (2/15) | 0.2902 | -0.0007 (2/15) |
| `LassoCV` | 0.2869 | -0.0040 (2/15) | 0.2900 | -0.0008 (2/15) |
| `RF-leaf50` | 0.2866 | -0.0043 (4/15) | 0.2866 | -0.0043 (1/15) |
| `ElasticNetCV` | 0.2863 | -0.0046 (2/15) | 0.2903 | -0.0006 (2/15) |
| `SVR-rbf` | 0.2858 | -0.0051 (2/15) | 0.2783 | -0.0126 (1/15) |
| `KRR-laplacian` | 0.2848 | -0.0061 (3/15) | 0.2803 | -0.0106 (2/15) |
| `SVR-linear` | 0.2847 | -0.0062 (2/15) | 0.2894 | -0.0015 (3/15) |
| `ET-leaf50` | 0.2847 | -0.0062 (2/15) | 0.2862 | -0.0047 (0/15) |
| `GP-RBF-ard` | 0.2830 | -0.0056 (4/10) | 0.2839 | -0.0047 (2/10) |
| `NuSVR-rbf` | 0.2830 | -0.0079 (3/15) | 0.2816 | -0.0093 (4/15) |
| `Spline+Ridge-rankfeat` | 0.2827 | -0.0082 (2/15) | 0.2860 | -0.0049 (1/15) |
| `ET-d4-leaf20` | 0.2820 | -0.0089 (0/15) | 0.2904 | -0.0005 (8/15) |
| `KRR-rbf-rankfeat` | 0.2810 | -0.0099 (2/15) | 0.2830 | -0.0079 (1/15) |
| `Ridge-rankfeat` | 0.2800 | -0.0109 (0/15) | 0.2917 | +0.0008 (11/15) |
| `PCA+Ridge` | 0.2793 | -0.0116 (0/15) | 0.2882 | -0.0027 (1/15) |
| `GP-RBF-ard-rankfeat` | 0.2788 | -0.0098 (2/10) | 0.2823 | -0.0064 (1/10) |
| `OMP-CV` | 0.2787 | -0.0122 (1/15) | 0.2864 | -0.0045 (0/15) |
| `Quantile-median` | 0.2780 | -0.0129 (0/15) | 0.2870 | -0.0039 (0/15) |
| `HGB-shallow` | 0.2780 | -0.0129 (0/15) | 0.2747 | -0.0162 (0/15) |
| `TheilSen` | 0.2777 | -0.0131 (0/15) | 0.2828 | -0.0081 (0/15) |
| `CatBoost-d3` | 0.2776 | -0.0133 (0/15) | 0.2918 | +0.0009 (7/15) |
| `CatBoost-d1` | 0.2769 | -0.0140 (0/15) | 0.2861 | -0.0048 (1/15) |
| `SGD-eps-insensitive` | 0.2767 | -0.0141 (0/15) | 0.2880 | -0.0029 (3/15) |
| `GBR-stumps` | 0.2736 | -0.0173 (0/15) | 0.2861 | -0.0048 (1/15) |
| `RANSAC-Ridge` | 0.2697 | -0.0212 (0/15) | 0.2888 | -0.0021 (4/15) |
| `SGD-huber` | 0.2621 | -0.0288 (0/15) | 0.2886 | -0.0023 (4/15) |
| `SVR-rbf-rankfeat` | 0.2565 | -0.0344 (0/15) | 0.2717 | -0.0192 (0/15) |
| `kNN-std-dist` | 0.2531 | -0.0378 (0/15) | 0.2918 | +0.0010 (10/15) |
| `kNN-rank-L1` | 0.2505 | -0.0404 (0/15) | 0.2885 | -0.0024 (10/15) |
| `kNN-rank-dist` | 0.2464 | -0.0445 (0/15) | 0.2887 | -0.0022 (10/15) |
| `kNN-PCA-rank` | 0.2429 | -0.0480 (0/15) | 0.2918 | +0.0010 (9/15) |
| `kNN-rank-unif` | 0.2404 | -0.0505 (0/15) | 0.2818 | -0.0091 (2/15) |
| `MLP-small` | 0.2402 | -0.0507 (0/15) | 0.2837 | -0.0072 (2/15) |
| `GP-RBF-iso` | 0.2350 | -0.0559 (0/15) | 0.2882 | -0.0027 (4/15) |
| `MLP-tiny-fixed` | 0.2257 | -0.0652 (0/15) | 0.2798 | -0.0111 (3/15) |


## 5. Confirmation on 30 fresh fold seeds (15..44)

Configurations fixed before running; none of these seeds was used for selection.

### 5a. Single models and FR/DE hybrids

fresh seeds 15..44 (n=30)   ridge-raw 0.2892 +/- 0.0060

| config | score | delta vs ridge | sd(delta) | wins/30 | FR rho | DE rho |
|---|---|---|---|---|---|---|
| C1-FRridge-DEspline | 0.2927 | +0.0035 | 0.0049 | 22/30 | 0.2079 | 0.3617 |
| C2-FRridge-DEkrr | 0.2930 | +0.0038 | 0.0043 | 25/30 | 0.2079 | 0.3639 |
| C3-FRhuber-DEspline | 0.2924 | +0.0032 | 0.0046 | 21/30 | 0.2111 | 0.3617 |
| C4-FRwinsorridge-DEspline | 0.2877 | -0.0014 | 0.0048 | 10/30 | 0.2096 | 0.3507 |
| hgb-shallow | 0.2631 | -0.0261 | 0.0116 | 1/30 | 0.1799 | 0.3352 |
| krr-both | 0.2834 | -0.0058 | 0.0089 | 7/30 | 0.1884 | 0.3639 |
| lgbm-leaves4 | 0.2746 | -0.0146 | 0.0083 | 2/30 | 0.1899 | 0.3459 |
| ridge-raw | 0.2892 | +0.0000 | 0.0000 | 0/30 | 0.2075 | 0.3577 |
| ridge-std | 0.2904 | +0.0013 | 0.0010 | 27/30 | 0.2079 | 0.3594 |
| ridge-winsor | 0.2914 | +0.0022 | 0.0020 | 25/30 | 0.2096 | 0.3603 |
| spline-both | 0.2863 | -0.0029 | 0.0055 | 11/30 | 0.1923 | 0.3617 |
| svr-rbf | 0.2769 | -0.0123 | 0.0082 | 2/30 | 0.2017 | 0.3452 |

### 5b. Blends, weights fixed in advance

fresh seeds 15..44   ridge-raw 0.2892 +/- 0.0060

| blend (weights fixed in advance) | score | delta vs ridge | sd(delta) | wins/30 |
|---|---|---|---|---|
| 0.60 ridge + 0.40 LGBM-leaves4 | 0.2951 | +0.0060 | 0.0035 | 28/30 |
| 0.60 ridge + 0.40 SVR-rbf | 0.2955 | +0.0064 | 0.0038 | 28/30 |
| 0.75 ridge + 0.25 HGB-shallow | 0.2954 | +0.0062 | 0.0028 | 30/30 |
| 0.60 ridge + 0.40 Spline+Ridge | 0.2925 | +0.0033 | 0.0023 | 28/30 |
| 0.60 ridge + 0.40 KRR-rbf | 0.2915 | +0.0023 | 0.0031 | 24/30 |
| equal ridge + LGBM + SVR-rbf | 0.2968 | +0.0076 | 0.0042 | 30/30 |
| 0.50 ridge + 0.25 LGBM + 0.25 SVR-rbf | 0.2981 | +0.0089 | 0.0031 | 30/30 |
| 0.50 ridge + .1667 LGBM + .1667 SVR + .1667 spline | 0.2980 | +0.0088 | 0.0025 | 30/30 |
| equal ridge + LGBM + SVR + spline | 0.2973 | +0.0081 | 0.0038 | 29/30 |

## 6. What the sweep actually says

**No family wins on its own, and the ordering is exactly the "capacity loses" ordering.**
Ranked by pooled score: robust/regularised linear (0.286-0.293) > additive splines,
kernel ridge and isotropic-Matern GPs (0.273-0.291) > constrained extra-trees
(0.285) > SVM (0.277) > boosting (0.263-0.282) > kNN (0.227-0.253) > GP with a fitted
RBF kernel (0.233) > MLP (0.215). The three models that edge above ridge --
`Ridge-winsor`, `Huber`, `PLS` -- do it by clipping, downweighting or projecting away
the tails. That is the same mechanism the rank transform of the target already exploits,
which is why the extra gain is only ~+0.002.

**Sparsity is actively harmful on Germany.** `LassoCV`, `ElasticNetCV`, `LassoLarsCV` and
`OMP-CV` all sit 0.005-0.017 below ridge, and the per-country table shows the entire loss
is on DE; FR is unaffected because FR's 3 features already *are* the sparse solution.
Germany's 29 features carry many small correlated contributions, and zeroing them costs
more than the variance it saves. `ARDRegression`, which shrinks smoothly rather than to
zero, is the one Bayesian variant that gains on DE (+0.0077, 13/15).

**Gaussian processes: the kernel choice is everything.** GPs got the most compute of any
family here (six kernels, marginal-likelihood fitted inside every fold) and the spread
between them is larger than the spread across all the other families combined.
`GP-linear` (DotProduct + White) reproduces ridge almost exactly -- 0.2904, corr 0.995 --
as it should, being Bayesian linear regression. `GP-Matern32` ties ridge (0.2913, +0.0004)
at corr 0.942, making it the cheapest source of mild diversity in the sweep. But
`GP-RBF-iso`, the same machinery with a *smooth* squared-exponential kernel, is among the
worst models tried (0.2327), and the per-country table shows exactly where it breaks:
DE rho collapses to 0.26 against ridge's 0.36 while FR is untouched. Marginal-likelihood
fitting picks a short lengthscale across Germany's 29 dimensions and interpolates noise.
Giving the same kernel per-dimension lengthscales (`GP-RBF-ard`) recovers most of that --
0.2327 -> 0.2779 -- because ARD can push the useless dimensions' lengthscales out to the
bound, but it still trails ridge by 0.011. `GP-RQ` (0.2866) and `GP-lin+RBF` (0.2849) land
in between. Conclusion: GPs are not unsuited to this data, smoothness assumptions are, and
the rough Matern 3/2 kernel is the only one that survives contact with Germany.

**Diversity is worth more than accuracy here.** The blend table inverts the standalone
ranking. `HGB-shallow` scores 0.2635 alone -- 0.027 *below* ridge, one of the worst
models tried -- and has the lowest OOF correlation with ridge of any usable model (0.776);
it is also the single most reliable blend partner found (**30/30 seeds** on fresh data).
`LGBM-leaves4` (0.2784 alone, corr 0.825) adds +0.0075. `SVR-rbf` (0.2772, corr 0.850)
adds +0.0064. Meanwhile `BayesianRidge` (0.2904 alone, corr 0.995) adds +0.0001. The
pairwise correlation matrix shows LightGBM / XGBoost / HistGB are 0.94-0.98 correlated
with each other, so they count as one member; SVR-rbf (0.81 to LightGBM) and
`kNN-std-dist` (0.75-0.81 to everything) are genuinely separate directions.

**Things that did not work, stated plainly.** Rank-transforming the *features* (as opposed
to the target) loses 0.010 -- the target transform is doing the work, and the features do
not need the same treatment. PCA before ridge loses 0.014: the leading components are not
the predictive ones. Quantile/LAD regression loses 0.013 and `TheilSen` 0.018 despite the
outlier story, because ranking the target has already dealt with the tails; RANSAC (-0.017)
and SGD-huber (-0.023) fail the same way. Every kNN variant loses 0.038-0.064. Winsorising
helps plain ridge (+0.0022) but *hurts* the spline model (`C4` at -0.0014 on fresh seeds) --
clipping and basis expansion do not combine.

**Caveats.** (a) The tree/boosting rows are *fixed* configurations scored directly on the
outer CV, so they are optimistic; they lost anyway. (b) The blend weights in section 3 are
scanned, so section 3 is selection-inflated -- section 5b is the number to quote.
(c) The ARD Gaussian processes use 10 seeds and a 50-iteration cap on the L-BFGS
marginal-likelihood optimiser; uncapped they were projected to need many hours on a
heavily loaded machine. (d) Everything here is fold-level CV on 851 FR / 643 DE rows;
a +0.006 CV gain is roughly one standard error of the leaderboard estimate at n=654.

## 7. Exact constructors worth keeping

```python
# ---- fold-internal preprocessing (fitted on the fold's training rows of ONE country)
med   = t[cols].median()                       # t = fold-train rows for this country
Xtr   = t[cols].fillna(med).to_numpy(float)
Xva   = v[cols].fillna(med).to_numpy(float)
mu, s = Xtr.mean(0), Xtr.std(0); s[s == 0] = 1.0
Xtr, Xva = (Xtr - mu) / s, (Xva - mu) / s      # "std" prep
ytr   = rankdata(t.TARGET.values) / len(t)     # rank target, within country, within fold
ALPHAS = np.logspace(-2, 4, 40)
cols   = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"] if country == "FR" else ALL_29
```

### 7.1 Best diversifier: heavily regularised LightGBM, blended (confirmed +0.0060, 28/30)

```python
import lightgbm as lgb            # RAW (unscaled) median-imputed features, rank target
lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=4,
                  min_child_samples=40, subsample=0.7, subsample_freq=1,
                  colsample_bytree=(1.0 if country == "FR" else 0.4),
                  reg_lambda=20.0, random_state=seed, n_jobs=1, verbose=-1)

z    = lambda v: (v - v.mean()) / v.std()      # global, NOT per country
pred = 0.6 * z(ridge_pred) + 0.4 * z(lgbm_pred)
```

### 7.2 Most reliable diversifier: HistGradientBoosting, blended (confirmed +0.0062, 30/30)

```python
from sklearn.ensemble import HistGradientBoostingRegressor   # RAW features
HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=300,
                              l2_regularization=10.0, min_samples_leaf=30,
                              early_stopping=False, random_state=seed)
pred = 0.75 * z(ridge_pred) + 0.25 * z(hgb_pred)
```

### 7.3 Best three-model ensemble (+0.0100, 15/15 on seeds 0..14)

```python
from sklearn.svm import SVR       # "std" prep, C/gamma tuned by 3-fold CV inside the fold
Tuned(SVR(kernel="rbf"), {"C": [1.0, 10.0, 100.0], "gamma": ["scale", 1e-2, 0.1]},
      seed=seed, n_inner=3)

pred = 0.50 * z(ridge_pred) + 0.25 * z(lgbm_pred) + 0.25 * z(svr_pred)
```

### 7.4 Best single model: ridge for France, mild nonlinearity for Germany (+0.0038, 25/30)

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

def make_model(country, seed):                 # "std" prep, rank target
    if country == "FR":
        return RidgeCV(alphas=ALPHAS)
    # DE option A: kernel ridge -- inner CV picks alpha=0.1, gamma=1e-3 in 17/20 folds
    return Tuned(KernelRidge(kernel="rbf"),
                 {"alpha": [0.1, 1.0, 10.0, 100.0], "gamma": [1e-3, 1e-2, 0.1]},
                 seed=seed, n_inner=3)
    # DE option B (+0.0035): additive cubic B-splines, inner CV picks n_knots=3 in 20/20
    return Tuned(make_pipeline(SplineTransformer(degree=3, extrapolation="linear"),
                               RidgeCV(alphas=ALPHAS)),
                 {"splinetransformer__n_knots": [3, 4, 6]}, seed=seed, n_inner=3)
```

### 7.5 Cheapest win: winsorise the features (+0.0022, 25/30). Model unchanged.

```python
lo, hi   = np.percentile(Xtr, 2, axis=0), np.percentile(Xtr, 98, axis=0)
Xtr, Xva = np.clip(Xtr, lo, hi), np.clip(Xva, lo, hi)   # then z-score, then RidgeCV
```

Do **not** combine 7.5 with 7.4 -- `C4` in section 5a shows the pair is worse than either.

### 7.6 The `Tuned` wrapper used above

```python
class Tuned(BaseEstimator, RegressorMixin):
    """Grid search by Spearman on an inner KFold, fitted entirely inside the outer fold."""
    def __init__(self, base, grid, seed=0, n_inner=3):
        self.base, self.grid, self.seed, self.n_inner = base, grid, seed, n_inner

    def fit(self, X, y):
        combos = [{}]
        for k in self.grid:
            combos = [dict(c, **{k: v}) for c in combos for v in self.grid[k]]
        cv = list(KFold(self.n_inner, shuffle=True, random_state=self.seed).split(X))
        best, self.best_params_ = -np.inf, combos[0]
        for params in combos:
            tot = 0.0
            for itr, iva in cv:
                m = clone(self.base).set_params(**params).fit(X[itr], y[itr])
                p = np.asarray(m.predict(X[iva])).ravel()
                tot += 0.0 if np.all(p == p[0]) else spearmanr(p, y[iva]).statistic
            if tot > best:
                best, self.best_params_ = tot, params
        self.best_ = clone(self.base).set_params(**self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.best_.predict(X)).ravel()
```

## 8. Out-of-fold bundle

`notes/oof_zoo.pkl` is a `dict[str, list[np.ndarray]]`: one entry per model, each a list of
15 out-of-fold prediction vectors (fold seeds 0..14, in order), each of length 1494 and
aligned to `P.build_frames()[0]` row order. `ridge_baseline` is included as the reference.

The five members were chosen for blend value subject to no two exceeding 0.95 mutual OOF
correlation: `LGBM-leaves4` (boosting), `SVR-rbf` (SVM), `HGB-shallow` (hist boosting,
lowest correlation with ridge at 0.776), `Spline+Ridge` (additive, best Germany model),
`kNN-std-dist` (instance-based, most distinct from everything else and the best France
model). Note `LGBM-leaves4` and `HGB-shallow` are 0.94 correlated with each other -- treat
them as alternatives, not as two independent members.

```python
import pickle, numpy as np
from scipy.stats import spearmanr
oof = pickle.load(open("notes/oof_zoo.pkl", "rb"))
z = lambda v: (v - v.mean()) / v.std()
blend = [0.5 * z(r) + 0.25 * z(l) + 0.25 * z(s)
         for r, l, s in zip(oof["ridge_baseline"], oof["LGBM-leaves4"], oof["SVR-rbf"])]
# -> 0.3009 pooled, +0.0100 vs ridge, 15/15 seeds
```
