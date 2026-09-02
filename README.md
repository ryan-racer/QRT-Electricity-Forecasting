# Explaining Electricity Prices

**QRT × ENS Data Challenge** — explain daily changes in French and German electricity futures from same-day weather, generation, consumption and commodity data.

![Spearman](https://img.shields.io/badge/Spearman-0.5658-2ea44f?style=flat-square)
![Rank](https://img.shields.io/badge/rank-6%20%2F%201157-1f6feb?style=flat-square)
![Benchmark](https://img.shields.io/badge/benchmark-0.1587-lightgrey?style=flat-square)
![Python](https://img.shields.io/badge/python-3.14-3776ab?style=flat-square)

Nearly all of the score came from understanding the data, not from picking models. The single largest step was noticing that the `ID` column, unlike `DAY_ID`, was never shuffled.

```bash
uv sync
uv run python src/make_submission.py   # → submissions/blend3_final.csv, byte-identical to the submitted file
```

## How the score was built

Pooled out-of-fold Spearman, day-grouped 5-fold, 15–20 randomised seeds. Each step was kept only if it beat the previous one on nearly every seed, compared on identical folds.

```
benchmark  (pooled OLS)        0.194  █████████████████
model each country separately  0.236  █████████████████████
rank-transform the target      0.290  █████████████████████████
recover the day ordering (ID)  0.524  ██████████████████████████████████████████████
cumulative returns, France     0.546  ████████████████████████████████████████████████
neighbouring days' targets     0.552  ████████████████████████████████████████████████
shrink France's spread         0.558  █████████████████████████████████████████████████
+ LightGBM                     0.569  ██████████████████████████████████████████████████
+ pooled CatBoost              0.572  ██████████████████████████████████████████████████
```

Final CV **0.5716 ± 0.0050** → public leaderboard **0.5658**. The harness was well calibrated.

## Six things about this data

**1 · `ID` encodes calendar order.** `DAY_ID` is genuinely shuffled — I tested it, found ~0 autocorrelation, and concluded no time structure existed. Wrong column. Sorted by `ID`, French nuclear output autocorrelates at **0.980** and consumption at **0.965**. Physical *levels* autocorrelate strongly while commodity *returns* sit near zero, which is how real daily data behaves and not something a shuffle can fake. France is `ID − 932`, Germany is `ID + 284`. The target is a price *change*, so first differences of fundamentals along that ordering are what drive it: the one-day change in German residual load alone correlates **0.76** with the German target. CV went from 0.29 to 0.52.

**2 · A day's French and German rows are bit-identical.** All 32 features match exactly; only `COUNTRY` and the target differ, and the two targets correlate at just **0.12**. A model without country can't tell them apart, which is exactly why the benchmark is capped where it is.

**3 · Least squares fights the metric.** The top 1% of days by |target| carry **31%** of squared-error weight but 1% of rank positions. Rank-transforming the target is worth +0.04 and costs nothing.

**4 · France is priced off levels, Germany off changes.** With time features Germany reached 0.77 within-country; France sat at 0.23 and nothing moved it. France is nuclear-heavy, so its marginal price is the *fuel cost* of the marginal thermal unit — a level. The commodity columns are already returns, so differencing them differences the wrong thing twice. Summing them (5- and 20-day trailing) reconstructs the level. France went 0.23 → 0.28; `CARBON_RET_cum5` is the strongest single French feature in the dataset.

**5 · The two countries should not be on one scale.** The metric ranks FR and DE in one pool and Germany is far better predicted. Halving France's spread around its own mean lets the reliable German predictions own the extremes. The opposite move — normalising both onto a common scale — loses **0.048**.

**6 · "Added capacity always loses" was true, then false.** Under the original 29 features every flexible model lost to ridge (gradient boosting: 0.21 vs 0.29). I took that as a property of the dataset. It was a property of the signal-to-noise ratio: once the time features existed, the *byte-identical* LightGBM config flipped from −0.013 to **+0.023** against ridge. France is the small version of the same lesson — a permutation test passes 9 of 29 features for Germany but only 3 for France, so France gets three base columns and Germany gets all of them.

## The model

Exactly what `src/make_submission.py` runs.

| features, per country | FR | DE |
|---|---:|---:|
| base columns (3 sign-flip duplicates dropped) | 3 | 29 |
| lag-1 difference + deviation from 7-day trailing mean, all 29 columns | 58 | 58 |
| 5- and 20-day cumulative sums of the three commodity returns | 6 | — |
| level / volatility / spread of the 3 nearest training days' targets | 6 | 6 |
| **total** | **73** | **93** |

| model | scope | settings |
|---|---|---|
| RidgeCV | per country | α by LOO over `logspace(−2, 4, 40)` |
| LightGBM | per country | `num_leaves=4 · n_estimators=400 · lr=0.05 · min_child_samples=20 · λ=1 · subsample=0.8 · colsample=0.7` |
| CatBoost | both countries, with a country flag | `depth=6 · iterations=400 · lr=0.05 · l2=6` |

All three fit the rank-transformed target with median imputation. Predictions are z-scored and blended

```
0.8 × ( 0.7·ridge + 0.3·lightgbm ) + 0.2 × catboost     →   0.56 / 0.24 / 0.20
```

then France's predictions are shrunk halfway toward their own mean. Every LightGBM weight from 0.2–0.4 won 20/20 seeds and CatBoost at 0.1–0.2 won 15/16; the weights used sit mid-plateau, not at the peaks. Trailing windows use `shift(1)` so a day never sees itself; neighbour features draw only on training targets and a day is never its own neighbour.

<details>
<summary><b>What didn't work</b> — recorded so nothing gets retried</summary>
<br>

FR−DE spread features · seasonal phase from solar · calendar features from the recovered ordering · missing-value indicators · row deletion · feature standardisation · feature rank transforms · gaussianised targets · sample weighting · robust losses · ElasticNet · partial pooling across countries · second differences · lagged levels · rolling feature volatility · EWMA deviations · longer cumulative windows · every fitted stack, gate or weight optimiser.

A per-row oracle choosing between ridge and LightGBM would score **0.68**, and about half of that is stable row-level structure — but any gate trained to learn it lands at chance accuracy. The two models' errors correlate at 0.61: knowing where one struggles says nothing about which to trust.

Time-series foundation models have nothing to find here either. With the ordering recovered, the target's own autocorrelation is two lags of mild mean reversion in Germany and nothing in France, which the neighbour-target features already capture.

</details>

## Layout

```
src/
  make_submission.py    the full pipeline, raw CSVs → submission
  qrt_timeorder.py      the ID ordering; difference and cumulative-return features
  qrt_temporal.py       neighbouring-day target features, with a leakage guard
  qrt_prep.py           loading, validation, randomised grouped folds, scoring
notebooks/
  eda.ipynb             exploratory analysis — discoveries 2 and 3 and the harness
  data_prep.ipynb       preprocessing decisions, each one measured
  benchmark_qrt_en.ipynb  the organisers' benchmark, unchanged
submissions/blend3_final.csv
data/raw/               the four challenge CSVs
```

The EDA notebook's §1.3 concludes no time structure exists — right about `DAY_ID`, wrong about the dataset. Discovery 1 came later; I've left the original reasoning in place with a note rather than rewriting it.
