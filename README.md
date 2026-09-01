# Electricity Forecasting Challenge

This repository contains the QRT ENS electricity price challenge materials, data, and benchmark notebook.

## Project Structure

- `data/raw/`: original challenge CSV files
- `docs/`: converted challenge statement and notes
- `notebooks/`: exploratory and benchmark notebooks
- `submissions/`: generated submission files

## Files

- `docs/challenge.md`: Markdown version of the original challenge PDF
- `notebooks/benchmark_qrt_en.ipynb`: benchmark notebook updated to use the local folder structure
- `notebooks/eda.ipynb`: exploratory data analysis — structure, target, missingness, feature/target association, and CV baselines
- `notebooks/data_prep.ipynb`: builds model-ready artifacts from the EDA findings
- `src/qrt_prep.py`: shared loading, validation, fold-safe transforms, CV harness and scoring
- `data/processed/`: `train.parquet`, `test.parquet`, `manifest.json`
- `data/raw/X_train.csv`: training input data
- `data/raw/y_train.csv`: training target data
- `data/raw/X_test_final.csv`: test input data
- `data/raw/y_test_random_final.csv`: random submission example

## Running the Benchmark

Run `uv sync`, then open `notebooks/benchmark_qrt_en.ipynb` and run the cells. The notebook reads data from `../data/raw/` and writes the benchmark submission to `../submissions/benchmark_qrt.csv`.

## EDA

`notebooks/eda.ipynb` — structure, target, missingness, feature space, signal, then modelling under a
grouped CV harness.

| model | pooled OOF Spearman |
|---|---|
| benchmark (pooled OLS, `COUNTRY` dropped) | 0.1937 |
| per-country ridge, 29 features | 0.2356 ± 0.0094 |
| + FR feature selection (3 features) | 0.2577 ± 0.0115 |
| + rank-transformed target | 0.2896 ± 0.0074 |
| + TabPFN v3 blend | 0.3005 ± 0.0067 |
| + ID time-ordering features | 0.5238 ± 0.0056 |
| + cumulative commodity returns (FR) | 0.5460 ± 0.0056 |
| + neighbour-target features (k=3) | 0.5523 ± 0.0070 |
| + FR prediction-spread shrink (α=0.5) | 0.5582 ± 0.0068 |
| **+ 30% LightGBM blend** | **0.5685 ± 0.0045** |

Main findings:

- A day's FR and DE rows are **bit-identical across all 32 features**; only `COUNTRY` and the target differ.
  The benchmark drops `COUNTRY`, so it cannot separate them.
- **`DAY_ID` is a decoy shuffle, but `ID` encodes true calendar order.** `t = ID % 1216` is an exact
  bijection onto the 1216 days (`ID 0..1215` chronological, `ID 1216..2147` the FR twin of `ID-1216`).
  Sorted by `ID`, `FR_NUCLEAR` autocorrelates at 0.980 and `FR_CONSUMPTION` at 0.965, against ~0.00
  under `DAY_ID`. Commodity *returns* stay near zero while *levels* do not — exactly as real data
  behaves. First differences and deviations from trailing means take CV from 0.289 to **0.524**.
- Season is recoverable from solar but carries no target signal.
- 6–8 days carry ~31% of the squared-error weight and ~1% of the rank positions, so least squares optimises
  the wrong thing. Rank-transforming the target is the single biggest win.
- Against a family-wise permutation null, DE gets 9 of 29 features through and FR only 3. France overfits
  badly on the full set; cutting it to 3 features more than triples FR within-country score.
- Everything that added capacity lost: FR−DE spreads, seasonal terms, gradient boosting, volatility scaling.

Run with the project venv (`uv sync`), selecting the `.venv` kernel in your editor.

## Data prep

`notebooks/data_prep.ipynb` turns the raw CSVs into `data/processed/`, encoding the EDA findings. Design
rule: anything *fitted* must be fittable inside a CV fold, so the saved data keeps its NaNs and the
transforms live in `src/qrt_prep.py` rather than being baked in.

```python
import sys; sys.path.insert(0, "src")
import qrt_prep as P

train, test, manifest = P.load_processed()
P.validate(train, test)                       # re-asserts every EDA invariant
folds = P.make_folds(train.DAY_ID, seed=0)    # grouped on DAY_ID, randomised
mean, sd, oof = P.cross_validate(train, fit_predict)
```

Settled by measurement, not convention: fold-median imputation (all options within noise),
rank-transformed target (+0.04, the main win), and permutation-null feature selection applied
**inside the fold**.

That last point corrects the EDA: selecting France's features once on the full training set inflated
its CV estimate by ~0.008. Honest reference is **0.2806 ± 0.0066** pooled OOF Spearman, not 0.2896.
The model itself was fine — only the self-assessment was contaminated.

## Final recipe

`submissions/blend_final.csv`. All numbers are pooled out-of-fold Spearman over 20 randomised
day-grouped fold seeds, compared **paired** on identical fold assignments.

```python
import sys; sys.path.insert(0, "src")
import qrt_prep as P, qrt_timeorder as TO, qrt_temporal as TT

train, test = P.build_frames()                       # drops 3 exact sign-flip columns
F = P.feature_columns(train)                         # 29 base features
tr, te, NEW = TO.add_time_features(train, test, F, lags=(1,), windows=(7,))
tr, te, CUM = TO.add_cumulative_returns(tr, te, windows=(5, 20))
# FR: 3 selected base + time + cumulative returns.  DE: all 29 base + time.
# Both: k=3 neighbour-target features, rank-transformed target.
# Predict, z-score, blend 0.7*ridge + 0.3*LightGBM, then shrink FR spread by 0.5.
```

Each component, and why:

| component | effect | reason |
|---|---|---|
| per-country models | +0.034 | a day's FR and DE rows are bit-identical apart from `COUNTRY` |
| rank-transformed target | +0.040 | 8 days carry 31% of squared-error weight and 1% of rank positions |
| `ID` time ordering | +0.234 | `ID` encodes true calendar order; the target is a price *change* |
| cumulative commodity returns (FR) | +0.022 | France is priced off fuel-cost *levels*, not daily changes |
| neighbour targets, k=3 | +0.006 | DE target mean-reverts (lag-1 −0.211) and its volatility clusters |
| FR spread shrink, α=0.5 | +0.006 | DE within-country is 0.767 vs FR 0.233; let DE own the pooled extremes |
| 30% LightGBM blend | +0.010 | OOF correlation 0.841 with ridge — diversity, not raw strength |

### Tested and rejected

Recorded so they are not retried: FR−DE spreads, seasonal/winter-ness terms, missing-value
indicators, row deletion, feature standardisation and centering, feature rank transforms,
gaussianised targets, ElasticNet, partial pooling, sample weighting, robust losses, a
differentiable soft-Spearman objective in JAX, rolling feature volatility, EWMA deviations,
calendar features built from the recovered position, and every *fitted* stack, gate or
weight-optimiser (all lose to a fixed blend).

Two rules that were true early and later became false, both worth remembering:
**"no time structure exists"** (true of `DAY_ID`, false of `ID`) and **"added capacity always
loses"** (true at 29 weak features, false once the time features exist — LightGBM flips from
−0.0125 to +0.0232).
