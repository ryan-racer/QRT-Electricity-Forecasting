# Feature-engineering search — QRT electricity price

Goal: find engineered features that raise the pooled OOF Spearman above the established
per-country RidgeCV baseline.

**Baseline (reproduced exactly):** per-country `RidgeCV(alphas=np.logspace(-2,4,40))` on the
rank-transformed target, FR = `[FR_WINDPOW, GAS_RET, CARBON_RET]`, DE = all 29 features.
15 seeds → **0.2909 ± 0.0069**. 30 seeds → 0.2893 ± 0.0068. 50 seeds → 0.2902 ± 0.0064.

**Headline:** one candidate clears the noise band. Adding two interaction terms to the *French*
model — `GAS_RET × CARBON_RET` and `GAS_RET × FR_WINDPOW` — gives a paired delta of
**+0.0071 (29/30 seeds)** at 30 seeds and **+0.0072 (49/50)** at 50 seeds. Germany gains nothing
from anything tried: 100+ DE candidates, best is +0.0034, i.e. noise.

---

## Protocol

* Every candidate is compared **paired** against the baseline on *identical* fold assignments
  (`P.make_folds(train.DAY_ID, seed=s)`), 30 seeds unless stated. Reported: mean paired delta,
  sd of the per-seed deltas, and win count.
* **Noise band: |delta| < 0.007.** Anything inside it is reported as noise regardless of how
  large its t-statistic looks. The paired t-statistics are large because the folds are shared —
  they measure fold-assignment noise, not sampling noise, and must not be read as evidence.
* Everything fitted is fitted **inside the fold on training rows only**: medians for imputation,
  means for centring, quantiles for hinges/winsorising, spline knots, PCA rotations, scalers.
* Harness: `scratchpad/harness.py` (numpy fast path) + `scratchpad/builders.py`. The winner was
  independently re-implemented against `P.cross_validate` in `scratchpad/b7_verify.py`; it
  reproduces 0.2893 → 0.2964 exactly, so the result is not a harness artefact.
* **Negative control:** adding one pure-Gaussian-noise column to FR costs −0.0019 (50 seeds);
  three cost −0.0046. So an added column carries an intrinsic penalty of roughly −0.002 each in
  this pipeline. A candidate must beat that before it is even neutral.

---

## The one win

```python
# Inside the fold, on the FR rows only, AFTER median imputation fitted on the training rows.
# mu is computed from the TRAINING fold and applied to both train and valid.
FR_FEATURES = ["FR_WINDPOW", "GAS_RET", "CARBON_RET"]

mu = X_train_fold[FR_FEATURES].mean()          # train-fold means, fold-safe

def add_fr_interactions(df, mu):
    d = df.copy()
    d["GASxCARBON"] = (d["GAS_RET"]   - mu["GAS_RET"]) * (d["CARBON_RET"] - mu["CARBON_RET"])
    d["GASxWIND"]   = (d["GAS_RET"]   - mu["GAS_RET"]) * (d["FR_WINDPOW"] - mu["FR_WINDPOW"])
    return d

FR_FEATURES_PLUS = FR_FEATURES + ["GASxCARBON", "GASxWIND"]
# Germany is left completely unchanged: all 29 raw features.
```

Centring is cosmetic — the uncentred product scores identically (+0.0047 vs +0.0047 for the
single term). Use the centred version anyway so the term is orthogonal-ish to the main effects.

### Evidence for it

| check | baseline | candidate | delta | detail |
|---|---|---|---|---|
| 30 seeds, 5-fold | 0.2893 | 0.2964 | **+0.0071** | 29/30 seeds |
| 50 seeds, 5-fold | 0.2902 | 0.2974 | **+0.0072** | 49/50 seeds |
| 30 seeds, 3-fold | 0.2834 | 0.2916 | +0.0082 | 28/30 |
| 30 seeds, 10-fold | 0.2932 | 0.3012 | +0.0080 | 30/30 |
| alphas `logspace(-3,6,60)` | 0.2891 | 0.2963 | +0.0072 | 29/30 |
| alphas `logspace(0,3,20)` | 0.2892 | 0.2963 | +0.0071 | 29/30 |
| independent re-implementation | 0.2893 | 0.2964 | +0.0071 | different code path |
| **40 disjoint half-samples** | 0.2673 | — | **+0.0059** (sd 0.0088) | 29/40 halves |
| FR forced onto all 29 features | 0.2793 | 0.2857 | +0.0064 | survives, but that config is far worse overall |
| OLS instead of RidgeCV (rank target) | 0.2918 | 0.2986 | +0.0068 | not a Ridge artefact |
| OLS / RidgeCV on the **raw** target | 0.2548 | 0.2490 | **−0.0059** | only works with the rank-transformed target |

The two terms are close to additive: alone they are +0.0047 (`GASxCARBON`, 29/30) and
+0.0020 (`GASxWIND`, 28/30); together +0.0071. The third possible pair,
`CARBON_RET × FR_WINDPOW`, is −0.0017 on its own and drags the full 3-pair set down to +0.0056.

### Why it is probably not a fishing artefact

The two winners come from the *smallest possible* search: interactions among the three
pre-registered FR features (only 3 such pairs exist). A separate wide scan of **52** interactions
between an FR commodity and each of the other 26 features — `GAS_RET × everything` and
`CARBON_RET × everything` — produced a best of **+0.0011**, with the whole distribution centred
near −0.0012. The winners are not the top of a long tail; the long tail is flat and slightly
negative.

### What the term actually is

Marginal Spearman of the products with the FR target is tiny (`GASxCARBON` +0.029,
`GASxWIND` +0.044 vs `CARBON_RET` +0.192), yet the fitted ridge coefficients are comparable to
the main effects (`CARBON_RET` +0.033, `GASxCARBON` +0.015, `GASxWIND` +0.014). These are partial,
not marginal, contributions.

Reading the signs:
* `GASxCARBON > 0` — French price responds **super-additively when gas and carbon rise together**.
  Both are inputs to the marginal thermal unit's cost, so a joint fuel-cost shock passes through
  more than the sum of two separate ones. `min(GAS_RET, CARBON_RET)` alone — no product at all —
  recovers +0.0040 of the +0.0047, and `max` recovers only +0.0011, which is exactly the "both
  must move" reading.
* `GASxWIND > 0` with `FR_WINDPOW` main effect < 0 — the depressing effect of French wind is
  **attenuated on days when gas returns are high**. When fuel costs are moving, the marginal unit
  sets the price and wind matters less.

**Caveat, stated plainly:** the gain lives partly in the tails of the product. Winsorising the two
commodity returns at 2% before multiplying halves the gain (+0.0023); rank-transforming them to
uniform before multiplying destroys it (−0.0011); signed-sqrt of the product keeps two thirds
(+0.0032). So a meaningful share of the effect comes from a minority of large-|return| days.
The 40 half-sample check (29/40 halves positive) argues it is not literally five days, but it is
not a uniformly-distributed effect either.

**Second caveat:** +0.007 is a real, repeatable effect *on these 1494 rows*. The sampling standard
error of a Spearman on the 654-row test set is roughly 1/√653 ≈ 0.039. A +0.007 improvement is
invisible at that scale. Take it as "free, and directionally right", not as a leaderboard move.

---

## Everything tested

156 unique candidates, all at 30 seeds, paired against the same baseline.
Verdict: `WIN` = delta ≥ +0.007, `noise` = |delta| < 0.007, `LOSS` = delta ≤ −0.007.

| candidate | pooled OOF | delta | sd | wins | t | verdict |
|---|---|---|---|---|---|---|
| `FR+ GASxCARBON+GASxWIND` | 0.2964 | +0.0071 | 0.0025 | 29/30 | +15.1 | WIN |
| `FR+ GASxCARBON+GASxWIND (rpt)` | 0.2964 | +0.0071 | 0.0025 | 29/30 | +15.1 | WIN |
| `FR+ GASxCARBON+GASxWIND+DExFR` | 0.2954 | +0.0062 | 0.0028 | 29/30 | +12.0 | noise |
| `FR+ pairs(3)` | 0.2949 | +0.0056 | 0.0028 | 27/30 | +10.7 | noise |
| `BOTH: GASxCARBON+GASxWIND FR & DE` | 0.2944 | +0.0051 | 0.0035 | 28/30 | +7.8 | noise |
| `FR+ GASxCARBON` | 0.2939 | +0.0047 | 0.0019 | 29/30 | +12.9 | noise |
| `DE+ sq(EU wind sum)` | 0.2926 | +0.0034 | 0.0016 | 28/30 | +11.1 | noise |
| `DE+ hinge hi+lo top3` | 0.2923 | +0.0030 | 0.0018 | 29/30 | +8.8 | noise |
| `FR+ GASxCARBON+CARBONxWIND` | 0.2917 | +0.0024 | 0.0023 | 26/30 | +5.8 | noise |
| `DE+ |NET_EXPORT-med|` | 0.2915 | +0.0023 | 0.0017 | 27/30 | +7.1 | noise |
| `DE+ ratio EUresid/EUcons` | 0.2916 | +0.0023 | 0.0013 | 30/30 | +9.8 | noise |
| `DE+ hinge hi+lo RESID` | 0.2915 | +0.0022 | 0.0014 | 27/30 | +8.5 | noise |
| `DE+ sq(WINDPOW)` | 0.2915 | +0.0022 | 0.0019 | 26/30 | +6.2 | noise |
| `DE+ DE_WINDPOWxFR_WINDPOW` | 0.2914 | +0.0021 | 0.0017 | 27/30 | +6.6 | noise |
| `FR+ GASxWIND` | 0.2913 | +0.0020 | 0.0013 | 28/30 | +8.1 | noise |
| `DE+ ratio RESID/CONS` | 0.2909 | +0.0016 | 0.0012 | 30/30 | +7.6 | noise |
| `DE+ GAS_RETxDE_WIND` | 0.2908 | +0.0015 | 0.0017 | 27/30 | +4.8 | noise |
| `DE+ spline df4 top3` | 0.2906 | +0.0013 | 0.0018 | 23/30 | +3.7 | noise |
| `DE+ |RESID-med|` | 0.2906 | +0.0013 | 0.0014 | 24/30 | +5.3 | noise |
| `BOTH: z-scale features` | 0.2904 | +0.0012 | 0.0010 | 28/30 | +6.4 | noise |
| `FR+ GAS_RETxDE_CONSUMPTION` | 0.2904 | +0.0011 | 0.0014 | 23/30 | +4.3 | noise |
| `FR+ CARBON_RETxDE_NUCLEAR` | 0.2903 | +0.0010 | 0.0012 | 24/30 | +4.6 | noise |
| `DE+ pca_sq(all,k=3)` | 0.2900 | +0.0008 | 0.0027 | 18/30 | +1.5 | noise |
| `DE+ RESIDxWINDPOW` | 0.2900 | +0.0007 | 0.0015 | 21/30 | +2.5 | noise |
| `FR+ GASxWIND+CARBONxWIND` | 0.2898 | +0.0006 | 0.0021 | 19/30 | +1.4 | noise |
| `FR: winsor2% all 3 (replace)` | 0.2898 | +0.0006 | 0.0007 | 23/30 | +4.2 | noise |
| `FR: winsor2% x3 (replace)` | 0.2898 | +0.0006 | 0.0007 | 23/30 | +4.2 | noise |
| `DE+ sq(EU resid sum)` | 0.2899 | +0.0006 | 0.0016 | 25/30 | +1.9 | noise |
| `DE+ spline df4 RESID` | 0.2898 | +0.0005 | 0.0010 | 25/30 | +2.7 | noise |
| `DE+ ratio WIND/RESID` | 0.2898 | +0.0005 | 0.0008 | 24/30 | +3.4 | noise |
| `DE+ sq(NET_EXPORT)` | 0.2897 | +0.0004 | 0.0019 | 17/30 | +1.1 | noise |
| `DE+ ratio RESID/THERMAL` | 0.2897 | +0.0004 | 0.0006 | 23/30 | +3.6 | noise |
| `DE+ all 5 ratios` | 0.2896 | +0.0003 | 0.0013 | 17/30 | +1.4 | noise |
| `DE+ ratio EUrenew/EUcons` | 0.2896 | +0.0003 | 0.0011 | 15/30 | +1.4 | noise |
| `DE+ sq(RESID)` | 0.2895 | +0.0002 | 0.0015 | 19/30 | +0.6 | noise |
| `FR: tanh all3 (replace)` | 0.2894 | +0.0001 | 0.0013 | 16/30 | +0.4 | noise |
| `FR: tanh x3 (replace)` | 0.2894 | +0.0001 | 0.0013 | 16/30 | +0.4 | noise |
| `DE+ ratio RENEW/CONS` | 0.2894 | +0.0001 | 0.0009 | 12/30 | +0.8 | noise |
| `FR+ CARBON_RETxFR_WIND` | 0.2893 | +0.0001 | 0.0012 | 20/30 | +0.3 | noise |
| `DE+ GAS_RETxRESID` | 0.2893 | +0.0000 | 0.0014 | 18/30 | +0.1 | noise |
| `DE+ meritorder x3` | 0.2893 | -0.0000 | 0.0027 | 13/30 | -0.0 | noise |
| `DE+ GAS_RETxDE_RESIDUAL_LOAD` | 0.2893 | +0.0000 | 0.0014 | 18/30 | +0.1 | noise |
| `FR+ GAS_RETxDE_COAL` | 0.2893 | +0.0000 | 0.0014 | 16/30 | +0.1 | noise |
| `FR+ DE_HYDRO (extra feat)` | 0.2892 | -0.0001 | 0.0014 | 15/30 | -0.5 | noise |
| `DE+ GAS_RETxDE_GAS` | 0.2892 | -0.0001 | 0.0011 | 16/30 | -0.6 | noise |
| `DE+ pairs(top4)` | 0.2891 | -0.0002 | 0.0028 | 15/30 | -0.4 | noise |
| `DE+ pctrank(RESID)` | 0.2891 | -0.0002 | 0.0004 | 8/30 | -2.3 | noise |
| `DE+ sq(DE_HYDRO)` | 0.2891 | -0.0002 | 0.0017 | 16/30 | -0.5 | noise |
| `DE+ hinge hi q90 RESID` | 0.2890 | -0.0003 | 0.0004 | 7/30 | -3.9 | noise |
| `DE+ GAS_RETxDE_NUCLEAR` | 0.2890 | -0.0003 | 0.0011 | 12/30 | -1.4 | noise |
| `DE+ ratio WINDPOW/WIND` | 0.2889 | -0.0004 | 0.0005 | 9/30 | -3.7 | noise |
| `FR+ GAS_RETxDE_LIGNITE` | 0.2889 | -0.0004 | 0.0014 | 15/30 | -1.6 | noise |
| `FR+ CARBON_RETxDE_CONSUMPTION` | 0.2889 | -0.0004 | 0.0007 | 9/30 | -3.2 | noise |
| `FR+ CARBON_RETxFR_NUCLEAR` | 0.2889 | -0.0004 | 0.0009 | 12/30 | -2.3 | noise |
| `FR+ CARBON_RETxDE_SOLAR` | 0.2889 | -0.0004 | 0.0015 | 13/30 | -1.4 | noise |
| `FR+ CARBON_RETxFR_COAL` | 0.2888 | -0.0005 | 0.0007 | 6/30 | -3.9 | noise |
| `DE+ GAS_RETxFR_RAIN` | 0.2887 | -0.0006 | 0.0012 | 10/30 | -2.7 | noise |
| `DE+ hinge hi q75 RESID` | 0.2886 | -0.0007 | 0.0006 | 2/30 | -6.2 | noise |
| `FR+ ratio EUrenew/EUcons` | 0.2886 | -0.0007 | 0.0009 | 9/30 | -4.2 | noise |
| `FR+ CARBON_RETxDE_WIND` | 0.2886 | -0.0007 | 0.0010 | 8/30 | -3.8 | noise |
| `FR+ GAS_RETxDE_SOLAR` | 0.2885 | -0.0008 | 0.0022 | 14/30 | -2.0 | noise |
| `FR+ GAS_RETxDE_RESIDUAL_LOAD` | 0.2885 | -0.0008 | 0.0013 | 8/30 | -3.4 | noise |
| `FR+ GAS_RETxFR_TEMP` | 0.2885 | -0.0008 | 0.0012 | 12/30 | -3.6 | noise |
| `FR+ ratio RENEW_FR/CONS_FR` | 0.2884 | -0.0009 | 0.0013 | 7/30 | -3.9 | noise |
| `FR+ ratio FRrenew/FRcons` | 0.2884 | -0.0009 | 0.0013 | 7/30 | -3.9 | noise |
| `DE+ RESIDxGAS` | 0.2883 | -0.0009 | 0.0013 | 8/30 | -4.0 | noise |
| `DE+ winsor2% commodities` | 0.2884 | -0.0009 | 0.0015 | 6/30 | -3.3 | noise |
| `FR+ GAS_RETxFR_CONSUMPTION` | 0.2884 | -0.0009 | 0.0013 | 8/30 | -3.7 | noise |
| `DE+ GAS_RETxDE_LIGNITE` | 0.2883 | -0.0010 | 0.0014 | 9/30 | -3.6 | noise |
| `FR+ CARBON_RETxDE_COAL` | 0.2883 | -0.0010 | 0.0012 | 8/30 | -4.5 | noise |
| `FR+ CARBON_RETxFR_RESIDUAL_LOAD` | 0.2883 | -0.0010 | 0.0014 | 6/30 | -3.9 | noise |
| `FR+ ratio WIND/RESID_FR` | 0.2882 | -0.0011 | 0.0011 | 2/30 | -5.2 | noise |
| `FR+ ratio FRWIND/FRRESID` | 0.2882 | -0.0011 | 0.0011 | 2/30 | -5.2 | noise |
| `DE+ CARBONxLIGNITE` | 0.2882 | -0.0011 | 0.0021 | 9/30 | -2.9 | noise |
| `DE+ ratio EXPORT/CONS` | 0.2882 | -0.0011 | 0.0008 | 2/30 | -7.2 | noise |
| `FR+ CARBON_RETxDE_LIGNITE` | 0.2882 | -0.0011 | 0.0012 | 7/30 | -5.0 | noise |
| `FR+ GAS_RETxFR_COAL` | 0.2882 | -0.0011 | 0.0009 | 2/30 | -6.8 | noise |
| `DE+ GAS_RETxDE_WINDPOW` | 0.2881 | -0.0012 | 0.0014 | 3/30 | -4.6 | noise |
| `FR+ CARBON_RETxFR_SOLAR` | 0.2881 | -0.0012 | 0.0012 | 4/30 | -5.5 | noise |
| `FR+ CARBON_RETxDE_RESIDUAL_LOAD` | 0.2880 | -0.0012 | 0.0011 | 3/30 | -5.9 | noise |
| `DE+ sq(DE_TEMP)` | 0.2879 | -0.0013 | 0.0014 | 5/30 | -5.3 | noise |
| `DE+ NUCLEARxRESID` | 0.2880 | -0.0013 | 0.0009 | 2/30 | -7.6 | noise |
| `FR+ GAS_RETxDE_RAIN` | 0.2880 | -0.0013 | 0.0012 | 4/30 | -6.2 | noise |
| `FR+ CARBON_RETxFR_CONSUMPTION` | 0.2880 | -0.0013 | 0.0012 | 4/30 | -5.5 | noise |
| `FR+ CARBON_RETxDE_GAS` | 0.2880 | -0.0013 | 0.0012 | 1/30 | -5.9 | noise |
| `FR+ CARBON_RETxFR_GAS` | 0.2880 | -0.0013 | 0.0015 | 4/30 | -4.8 | noise |
| `FR+ CARBON_RETxFR_TEMP` | 0.2880 | -0.0013 | 0.0015 | 7/30 | -4.5 | noise |
| `DE+ GAS_RETxDE_RAIN` | 0.2878 | -0.0014 | 0.0012 | 3/30 | -6.5 | noise |
| `FR+ GAS_RETxDE_NET_EXPORT` | 0.2879 | -0.0014 | 0.0011 | 3/30 | -6.4 | noise |
| `FR+ GAS_RETxDE_GAS` | 0.2879 | -0.0014 | 0.0015 | 4/30 | -4.9 | noise |
| `FR+ GAS_RETxDE_WINDPOW` | 0.2878 | -0.0014 | 0.0016 | 2/30 | -4.9 | noise |
| `FR+ GAS_RETxFR_RESIDUAL_LOAD` | 0.2878 | -0.0014 | 0.0014 | 8/30 | -5.5 | noise |
| `FR+ GAS_RETxFR_WIND` | 0.2879 | -0.0014 | 0.0019 | 10/30 | -4.0 | noise |
| `FR+ CARBON_RETxDE_FR_EXCHANGE` | 0.2879 | -0.0014 | 0.0014 | 4/30 | -5.4 | noise |
| `DE+ RESIDxNET_EXPORT` | 0.2878 | -0.0015 | 0.0016 | 5/30 | -4.8 | noise |
| `FR+ GAS_RETxDE_NUCLEAR` | 0.2878 | -0.0015 | 0.0015 | 6/30 | -5.5 | noise |
| `FR+ CARBON_RETxDE_TEMP` | 0.2877 | -0.0015 | 0.0013 | 0/30 | -6.6 | noise |
| `FR+ DE_NET_EXPORT (extra)` | 0.2877 | -0.0016 | 0.0016 | 5/30 | -5.3 | noise |
| `DE+ EXCHANGExRESID` | 0.2876 | -0.0016 | 0.0015 | 3/30 | -6.0 | noise |
| `FR+ GAS_RETxFR_GAS` | 0.2877 | -0.0016 | 0.0013 | 2/30 | -6.2 | noise |
| `FR+ GAS_RETxFR_HYDRO` | 0.2877 | -0.0016 | 0.0012 | 2/30 | -7.2 | noise |
| `FR+ GAS_RETxDE_WIND` | 0.2877 | -0.0016 | 0.0022 | 5/30 | -4.0 | noise |
| `FR+ GAS_RETxDE_TEMP` | 0.2877 | -0.0016 | 0.0013 | 0/30 | -6.5 | noise |
| `FR+ CARBON_RETxFR_NET_EXPORT` | 0.2877 | -0.0016 | 0.0012 | 1/30 | -7.1 | noise |
| `FR+ CARBON_RETxFR_HYDRO` | 0.2877 | -0.0016 | 0.0018 | 3/30 | -4.9 | noise |
| `FR+ CARBON_RETxDE_WINDPOW` | 0.2877 | -0.0016 | 0.0015 | 2/30 | -5.8 | noise |
| `FR+ CARBONxWIND` | 0.2876 | -0.0017 | 0.0015 | 2/30 | -6.1 | noise |
| `DE+ GAS_RETxDE_FR_EXCHANGE` | 0.2875 | -0.0017 | 0.0017 | 6/30 | -5.4 | noise |
| `FR+ GAS_RETxDE_FR_EXCHANGE` | 0.2876 | -0.0017 | 0.0014 | 2/30 | -6.4 | noise |
| `FR+ GAS_RETxFR_SOLAR` | 0.2875 | -0.0017 | 0.0016 | 2/30 | -5.9 | noise |
| `FR+ CARBON_RETxDE_RAIN` | 0.2876 | -0.0017 | 0.0013 | 1/30 | -6.8 | noise |
| `FR+ quad(3)` | 0.2875 | -0.0018 | 0.0043 | 12/30 | -2.2 | noise |
| `FR+ hinge hi q75 x3` | 0.2875 | -0.0018 | 0.0013 | 1/30 | -7.2 | noise |
| `DE+ |DE_FR_EXCHANGE|` | 0.2874 | -0.0018 | 0.0010 | 1/30 | -9.4 | noise |
| `FR+ GAS_RETxDE_HYDRO` | 0.2875 | -0.0018 | 0.0015 | 2/30 | -6.6 | noise |
| `FR+ CARBON_RETxDE_NET_EXPORT` | 0.2875 | -0.0018 | 0.0010 | 0/30 | -9.4 | noise |
| `FR+ COAL_RET (extra feat)` | 0.2874 | -0.0019 | 0.0019 | 5/30 | -5.5 | noise |
| `DE+ sq(GAS)` | 0.2874 | -0.0019 | 0.0012 | 3/30 | -8.0 | noise |
| `FR+ GAS_RETxFR_NET_EXPORT` | 0.2874 | -0.0019 | 0.0012 | 0/30 | -8.6 | noise |
| `FR+ CARBON_RETxDE_HYDRO` | 0.2874 | -0.0019 | 0.0014 | 0/30 | -7.5 | noise |
| `DE+ squares(top6)` | 0.2872 | -0.0020 | 0.0028 | 7/30 | -4.0 | noise |
| `FR+ CARBON_RETxFR_RAIN` | 0.2873 | -0.0020 | 0.0017 | 1/30 | -6.3 | noise |
| `FR+ GAS_RETxFR_NUCLEAR` | 0.2871 | -0.0021 | 0.0017 | 0/30 | -6.9 | noise |
| `FR+ CARBON_RETxCOAL_RET` | 0.2872 | -0.0021 | 0.0013 | 0/30 | -8.5 | noise |
| `FR+ FRWINDxDEWIND` | 0.2871 | -0.0022 | 0.0015 | 0/30 | -7.9 | noise |
| `DE+ GAS_RETxFR_SOLAR` | 0.2871 | -0.0022 | 0.0010 | 1/30 | -11.6 | noise |
| `FR+ GAS_RETxCOAL_RET` | 0.2871 | -0.0022 | 0.0016 | 0/30 | -7.2 | noise |
| `FR+ pctrank x3` | 0.2870 | -0.0023 | 0.0018 | 4/30 | -6.7 | noise |
| `DE+ HYDROxRESID` | 0.2869 | -0.0023 | 0.0019 | 2/30 | -6.6 | noise |
| `FR+ GAS_RETxFR_RAIN` | 0.2869 | -0.0024 | 0.0013 | 0/30 | -10.1 | noise |
| `FR+ hinge lo q25 x3` | 0.2867 | -0.0025 | 0.0024 | 1/30 | -5.6 | noise |
| `FR+ |GAS|,|CARBON|` | 0.2867 | -0.0025 | 0.0016 | 1/30 | -8.3 | noise |
| `FR+ winsor2% commodities` | 0.2868 | -0.0025 | 0.0024 | 5/30 | -5.7 | noise |
| `DE+ GAS_RETxFR_NUCLEAR` | 0.2868 | -0.0025 | 0.0014 | 0/30 | -9.3 | noise |
| `DE+ GAS_RETxFR_NET_EXPORT` | 0.2867 | -0.0026 | 0.0012 | 0/30 | -11.9 | noise |
| `DE+ GAS_RETxFR_COAL` | 0.2866 | -0.0027 | 0.0015 | 1/30 | -9.4 | noise |
| `DE+ GAS_RETxDE_HYDRO` | 0.2866 | -0.0027 | 0.0014 | 0/30 | -10.1 | noise |
| `DE+ GAS_RETxDE_COAL` | 0.2864 | -0.0028 | 0.0015 | 0/30 | -10.2 | noise |
| `DE+ GAS_RETxDE_NET_EXPORT` | 0.2862 | -0.0030 | 0.0017 | 2/30 | -9.9 | noise |
| `DE+ GAS_RETxFR_WINDPOW` | 0.2863 | -0.0030 | 0.0013 | 0/30 | -12.9 | noise |
| `FR: signed-sqrt all3 (repl)` | 0.2860 | -0.0032 | 0.0016 | 1/30 | -11.2 | noise |
| `FR: signed-sqrt x3 (replace)` | 0.2860 | -0.0032 | 0.0016 | 1/30 | -11.2 | noise |
| `DE+ CARBONxCOAL` | 0.2860 | -0.0033 | 0.0016 | 0/30 | -11.2 | noise |
| `FR+ hinge hi+lo x3` | 0.2858 | -0.0035 | 0.0026 | 0/30 | -7.1 | noise |
| `FR+ spline df6 x3` | 0.2856 | -0.0037 | 0.0045 | 2/30 | -4.5 | noise |
| `DE+ sq(DE_TEMP)+sq(FR_TEMP)` | 0.2855 | -0.0037 | 0.0015 | 0/30 | -13.6 | noise |
| `DE+ GAS_RETxDE_SOLAR` | 0.2848 | -0.0044 | 0.0014 | 0/30 | -17.0 | noise |
| `DE+ GAS_RETxFR_HYDRO` | 0.2847 | -0.0045 | 0.0025 | 0/30 | -9.7 | noise |
| `DE+ GAS_RETxFR_GAS` | 0.2846 | -0.0047 | 0.0017 | 0/30 | -15.2 | noise |
| `DE+ GAS_RETxDE_CONSUMPTION` | 0.2842 | -0.0050 | 0.0015 | 0/30 | -17.6 | noise |
| `DE+ GAS_RETxFR_RESIDUAL_LOAD` | 0.2841 | -0.0051 | 0.0020 | 0/30 | -14.1 | noise |
| `DE+ pairs(top6)` | 0.2835 | -0.0057 | 0.0047 | 3/30 | -6.5 | noise |
| `DE+ GAS_RETxFR_CONSUMPTION` | 0.2832 | -0.0060 | 0.0019 | 0/30 | -17.1 | noise |
| `FR+ cube x3` | 0.2826 | -0.0067 | 0.0026 | 0/30 | -14.1 | noise |
| `FR+ squares(3)` | 0.2818 | -0.0075 | 0.0024 | 0/30 | -16.7 | LOSS |
| `DE+ commodity x top3 (9)` | 0.2815 | -0.0077 | 0.0035 | 0/30 | -11.8 | LOSS |

### Naming key

* `FR+ X` — X added to the 3-feature French model, Germany untouched.
* `DE+ X` — X added to the 29-feature German model, France untouched.
* `sq(v)` = `(v − mean_train(v))²`; `cube` likewise; `|v−med|` = distance from the train median.
* `AxB` = `(A − mean_train A)(B − mean_train B)`.
* `pairs(S)` = all pairwise products within S; `squares(S)` = all squares; `quad` = both.
* `hinge hi qN(v)` = `max(v − quantile_N(v_train), 0)`; `lo` is the mirror.
* `spline dfK(v)` = `SplineTransformer(n_knots=K, degree=3, knots="quantile")` fitted on the fold.
* `ratio A/B` = `ΣA / (ΣB + s)` with `s = −min(ΣB_train) + 0.5`, so the denominator is positive
  on the training fold (the columns are z-scores, so a naive ratio is undefined).
* `pctrank(v)` = empirical CDF position of v in the training fold.
* `top3/top4/top6` for DE = `DE_RESIDUAL_LOAD, DE_NET_EXPORT, DE_WINDPOW, DE_GAS, DE_HYDRO, FR_WINDPOW`.

---

## Mechanism probe: what form does the FR commodity interaction want?

All variants replace `GASxCARBON` only; 30 seeds, paired.

| form | delta | wins |
|---|---|---|
| centred product `(G−ḡ)(C−c̄)` | +0.0047 | 29/30 |
| uncentred product `G·C` | +0.0047 | 29/30 |
| `min(GAS_RET, CARBON_RET)` | +0.0040 | 29/30 |
| signed-sqrt of the product | +0.0032 | 29/30 |
| `min` + `max` | +0.0026 | 27/30 |
| product of 2%-winsorised inputs | +0.0023 | 28/30 |
| `max(GAS_RET, CARBON_RET)` | +0.0011 | 21/30 |
| `relu(G)·relu(C)` | +0.0009 | 24/30 |
| product of rank-uniform inputs | −0.0011 | 9/30 |

with `GASxWIND` also added: `min+max` +0.0047, winsorised product +0.0044, rank product +0.0002 —
all below the raw-product version's +0.0071.

Reading: the signal is "**both commodities moved, and moved a lot**". Rank-transforming the inputs
kills it because it removes magnitude; `min` keeps most of it because it fires only when both are up.

---

## Learner / target sensitivity of the winner

| pipeline | baseline | with the 2 FR interactions | delta |
|---|---|---|---|
| RidgeCV, rank target (the baseline) | 0.2893 | 0.2964 | +0.0071 |
| OLS, rank target | 0.2918 | 0.2986 | +0.0068 |
| Huber, rank target | 0.2921 | 0.2972 | +0.0050 |
| OLS, raw target | 0.2548 | 0.2490 | −0.0059 |
| RidgeCV, raw target | 0.2539 | 0.2485 | −0.0055 |

The interaction only helps once the target's tails are suppressed. On the raw target the fit is
already dominated by a handful of extreme days, and a heavy-tailed product feature makes that worse.
(Incidental: plain OLS edges RidgeCV on the baseline, 0.2918 vs 0.2893. Not chased — outside brief.)

---

## Negative results worth recording

**Germany gains from nothing.** Across ~100 DE candidates — squares, cubes, hinges at q25/q75/q90,
quantile splines, pairwise products of the top 4 and top 6 features, the full 9-way
commodity × top-3 interaction block, merit-order terms (`CARBON_RET × DE_LIGNITE`,
`CARBON_RET × DE_COAL`, `GAS_RET × DE_RESIDUAL_LOAD`), physically-motivated ratios
(residual-load/consumption, renewables/consumption, net-export/consumption,
residual-load/thermal-capacity, windpower/wind, pan-European versions of all of them),
percentile-rank tightness, distance-from-median, squared PCA scores, and a 29-wide
`GAS_RET × everything` scan — the best result is `sq(EU wind sum)` at **+0.0034** and the median is
about −0.0010. Nothing approaches the noise band. This is consistent with the EDA's finding that
Germany's relationships are already close to linear and that *this dataset punishes added capacity*.

**Physical ratios do not transfer.** Every feature is pre-standardised to mean 0 / sd 1, so a ratio
of two columns is a ratio of z-scores, not of megawatts. "Renewables share of consumption" and
"residual load over thermal capacity" are not recoverable from the released data — the shift needed
to make the denominator positive is arbitrary, and the resulting feature has no physical meaning.
Best of the eight ratios tried: +0.0023 (`EUresid/EUcons`). Treat this family as closed.

**Nonlinear transforms of single strong features lose.** FR squares −0.0075, FR cubes −0.0067,
FR hinges −0.0018 to −0.0035, FR degree-3 splines (df 6) −0.0037, FR percentile ranks −0.0023.
DE equivalents all land inside noise. The one-added-noise-column control (−0.0019) explains a large
share of these: they are mostly paying the capacity tax and buying nothing.

**Monotone re-shaping of the FR inputs is neutral.** Winsorising all three at 2% +0.0006,
tanh +0.0001, signed-sqrt −0.0032. The rank-transform win in the EDA belongs to the *target*, not
the features.

**Per-country standardisation of the design matrix: +0.0012, noise.** Confirms the earlier finding.
RidgeCV's alpha search absorbs a global rescaling; the only thing standardisation changes is the
*relative* penalty across columns, and here that does not matter.

**Adding a 4th feature to France still loses.** `DE_HYDRO` −0.0001, `DE_NET_EXPORT` −0.0016,
`COAL_RET` −0.0019. The 3-feature rule survives; what France wants is *nonlinearity in the three*,
not more columns.

**Post-hoc per-country calibration of the predictions hurts** (not feature engineering, but tested
because the metric pools the two countries). Rank-normalising each country's OOF predictions to
uniform before pooling: **−0.0129** (0/30). Z-scoring them: **−0.0080** (0/30). The natural spread
of the two ridge outputs already carries cross-country information — Germany's predictions are
genuinely more confident, and flattening that destroys the pooled ordering.

**Applying the FR interactions to Germany as well is worse than FR-only** (+0.0051 vs +0.0071),
as is adding a third `GAS_RET × DE_WINDPOW` term to France (+0.0062).

---

## Reproducing

Scripts live in the session scratchpad
(`.../scratchpad/`): `harness.py` (paired runner), `builders.py` (fold-safe feature builders),
`bfr.py`/`bfr2.py`/`bde.py`/`b3.py` (candidate batches), `b8_scan.py`/`b9_scan_de.py` (wide
interaction scans), `b5_robust.py`/`b12_robust2.py` (fold-count, seed-count, alpha-grid
robustness + noise controls), `b10_stability.py` (40 disjoint half-samples),
`b11_form.py` (interaction form), `b13_learner.py` (learner/target sensitivity),
`b7_verify.py` (independent re-implementation), `b6_diag.py` (coefficients and marginal rhos).
Only the code block under "The one win" is needed to reproduce the improvement.
