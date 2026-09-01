# QRT / ENS 2023 "Can you explain the price of electricity?" — winner's talk

## Source

**Primary:** Collège de France, chaire *Sciences des données* (Stéphane Mallat), seminar of
**31 January 2024**, *"Présentations des gagnants des challenges 2023"*.

- Video: <https://www.youtube.com/watch?v=Xd4ESBwFn5s> (54:08, title
  *"Apprentissage et génération par échantillonnage aléatoire (6) — Stéphane Mallat (2023-2024)"*)
- Event page (EN): <https://www.college-de-france.fr/en/agenda/seminar/learning-and-generation-by-random-sampling/presentations-of-the-2023-challenge-winners>
- Raw audio (used for re-transcription): <https://podcastfichiers.college-de-france.fr/mallat-sem-20240131.m4a>
- Winners page: <https://challengedata.ens.fr/winners/2023> — QRT challenge:
  **1. Mohamed Ali Ben Amara, 2. "mb", 3. Amine Abdelkader**

**Transcript status: OBTAINED (and cross-verified with a second ASR engine).** French auto-generated captions were pulled from YouTube
(`youtube_transcript_api Xd4ESBwFn5s --languages fr`); yt-dlp and the raw `timedtext` endpoint both
failed (PO-token gating), the caption API did not. The talk is in **French**; the QRT data provider's
introduction immediately before it is in **English**. What follows is verbatim ASR (lightly
de-stuttered where marked) plus an English translation.

The relevant block is **07:24 – 13:00** of the video. There are **no slides published** by the
Collège de France, no GitHub/GitLab repo, and no blog writeup by Ben Amara that I could find
(his only public GitLab project, `Dalyamara/research-internship`, is unrelated). So the video is the
only first-hand account.

---

## Part 1 — QRT's own introduction (Eduardo Benetti, QRT), 04:35–07:24, in English

Verbatim, ASR-cleaned:

> "…electricity prices — it was not about predicting the price of the electricity in the future but
> **explaining what was going on today with information that we have today**. As a benchmark for the
> score: if you knew all this information you could get 100% prediction on the price — you know, if
> you know the prices of all the commodities, of all the production in France, all the production in
> Germany, you could predict this perfectly. So the best score would have been 100%. […] We use data
> which is weather data, data production in France from nuclear, wind, coal, both in Germany also.
> And it was a challenging problem because each of the countries have very different behaviours.
>
> "…something that we saw: in the beginning scores were around 25% correlation, 27. And **suddenly we
> saw a big jump in scores**. Something that we noticed was that **there was a leak in the data, in
> which we had anonymised the data in a way that you couldn't tell the dates — and you could
> reproduce, you could sort the test set, and that would give you some information.**
>
> "Something that was very interesting after looking at many of the solutions from students is that
> **using this leak was not necessary to get one of the top scores** — you could really use arguments
> both fundamental, using a real knowledge of the electricity market, or understanding the
> seasonality of electricity through time, to do this in a non-leak way. So we saw really some
> brilliant solutions, both using just machine learning, using fundamental arguments, using physics
> arguments for the seasonality. […] **We hired a couple of people that participated in this
> challenge** and gave us a very interesting solution."

Key points: QRT **confirms the leak exists and confirms it is a sortability leak on the test set**
(recovering the temporal ordering), and confirms it caused a visible jump from ~0.25–0.27 to higher
scores. They also state the leak was *not required* to reach a top score.

---

## Part 2 — Mohamed Ali Ben Amara (1st place), 07:24–13:00, in French

### Verbatim French (YouTube ASR, de-stuttered; ASR artefacts marked [sic])

> "Bonjour, je suis Mohamed Ali Ben Amara, je suis étudiant 3e année à l'ENSAE [sic: "l'in"] Paris et
> le M2 MVA, et je vais vous présenter ma solution pour le challenge.
>
> "Le but est d'expliquer ou de prédire les variations quotidiennes des prix de l'électricité pour la
> France et l'Allemagne. Comme variables explicatives on a des données météorologiques quotidiennes —
> la température, la pluie, le vent — et le prix des produits énergétiques, et la consommation comme
> le photovoltaïque et le nucléaire. Et on a aussi des informations sur l'utilisation au quotidien de
> l'électricité comme la consommation, et les échanges, les imports-exports entre les deux pays.
>
> "Donc on veut prédire les variations quotidiennes, mais on ne veut pas juste minimiser une erreur
> quadratique — MSE ou MAE normal — mais on veut **maximiser la corrélation de Spearman**.
>
> "Les idées principales de la solution sont juste dans **la création des variables, le feature
> engineering**, et **un petit changement que j'ai fait sur le target**.
>
> "Alors, pour la création des features, on peut voir que les données de consommation et les données
> météorologiques **ne sont pas stationnaires** — la température pendant l'été, c'est pas vraiment la
> même distribution que la température pendant l'hiver. Et donc le premier challenge était d'enlever
> cette non-stationnarité. Donc ce que j'ai fait, c'est d'essayer de **créer des features basés sur
> des clusters** : je crée plusieurs clusters en se basant à chaque fois sur un groupe de features,
> et après **je différencie par soit la moyenne ou la médiane de ce cluster** pour avoir un peu de
> stationnarité **et reconstruire un peu l'aspect temps**. Par exemple, si j'arrive à reconstruire un
> cluster qui groupe les dates de l'été et un autre pour l'hiver, je peux différencier la
> consommation ou la température de ce jour-là par la moyenne en juillet ou pendant l'été, et c'est
> un peu plus stationnaire.
>
> "Et cette idée m'a [donné] — j'ai eu **une solution qui a 0.32 de score**, donc dans le top 10 ou le
> top 12, **sans leak**, qui est juste en se basant sur cette création de features et un petit
> changement sur le target.
>
> "**Mais j'ai aussi trouvé une correspondance entre les ID — genre le feature ID — et les clusters.
> Et donc j'ai fait un reclassement par ID, et quand j'ai fait les plots, j'ai trouvé que je peux
> reconstruire l'aspect temps : genre, je trouve une time series.** Donc ce que j'ai fait, **j'ai
> différencié les bases [sic], et après j'ai fait une différenciation par des moyennes mouvantes
> [mobiles] sur les valeurs passées**. Et ce sont les features principales qui m'ont aidé pour faire
> le jump.
>
> "Et après, pour changer le target : la corrélation de Spearman vérifie principalement **les rangs**
> des prédictions — il faut avoir une relation monotone entre ce qu'on prédit et ce qu'on a. Par
> contre la plupart des algorithmes optimisent soit l'erreur quadratique ou absolue, genre MSE ou MAE.
> Et même de petits changements dans le MSE — par exemple dans le premier exemple le MSE est très
> faible mais on a une corrélation de Spearman de 0.5, car la petite erreur entre 0.1 et 0.12 et 0.11,
> [même] prédite comme il faut, on a **un flip dans les rangs** et ça va induire une corrélation de
> Spearman de 0.5. Donc ce que j'ai fait : **j'ai remplacé les targets par leur rang**. Si on a 0.1,
> 0.11 et 0.2, on peut le remplacer par 1, 2, 3 — et même si on a un MSE plus grand, la corrélation de
> Spearman est meilleure. Ce sont deux exemples un peu extrêmes, mais c'est l'idée qui m'a aussi aidé
> à faire un très bon jump.
>
> "Et pour l'algorithme : juste un petit point là-dessus aussi, il y a plusieurs variations de **soft
> ranking**, des implémentations différentiables de la corrélation de Spearman — j'ai expérimenté avec
> PyTorch [et] XGBoost, mais c'est **pas très stable** et c'est du **sur-apprentissage en direct après
> deux-trois itérations**.
>
> "Pour l'algorithme j'ai essayé plusieurs modèles linéaires — régression linéaire, des LASSO — et des
> modèles sur les arbres, et finalement **j'ai décidé d'utiliser principalement le CatBoost, car il
> est très bien avec les données catégoriques** et **la feature la plus importante est la feature de
> country** — genre Allemagne ou France. Et après j'ai fait un petit **feature selection à partir du
> CatBoost que j'ai entraîné**, et **je l'ai entraîné sur toute la dataset**. C'est tout, merci."

### English translation

> "Hello, I'm Mohamed Ali Ben Amara, a third-year student at ENSAE Paris and on the MVA master's, and
> I'll present my solution for the challenge.
>
> "The goal is to explain, or predict, the daily variations of electricity prices for France and
> Germany. As explanatory variables we have daily weather data — temperature, rain, wind — plus the
> price of energy commodities, and production such as photovoltaic and nuclear. We also have
> information on daily electricity use: consumption, exchanges, and imports/exports between the two
> countries.
>
> "So we want to predict daily variations, but we don't just want to minimise a squared error — plain
> MSE or MAE — we want to **maximise Spearman correlation**.
>
> "The main ideas of the solution are entirely in **feature construction / feature engineering**, plus
> **one small change I made to the target**.
>
> "For feature construction: you can see that the consumption data and the weather data **are not
> stationary** — temperature in summer isn't really the same distribution as temperature in winter. So
> the first challenge was to remove this non-stationarity. What I did was **build cluster-based
> features**: I create several clusterings, each based on a group of features, and then **I difference
> [each observation] by either the mean or the median of its cluster**, to get some stationarity **and
> to reconstruct a bit of the time aspect**. For example, if I manage to reconstruct a cluster that
> groups the summer dates and another for winter, I can difference that day's consumption or
> temperature by the July / summer mean, and that's a bit more stationary.
>
> "That idea gave me **a solution scoring 0.32, i.e. top 10 or top 12, WITHOUT the leak** — just from
> this feature construction plus the small change to the target.
>
> "**But I also found a correspondence between the IDs — the ID feature — and the clusters. So I
> re-sorted by ID, and when I made the plots I found that I could reconstruct the time aspect: I get a
> time series.** So what I did was **difference [the series], and then take differences against moving
> averages of the past values**. Those are the main features that let me make the jump.
>
> "Then, on changing the target: Spearman correlation only checks **the ranks** of the predictions —
> you need a monotone relation between what you predict and what you have. But most algorithms
> optimise squared or absolute error, MSE or MAE. Even tiny changes in MSE… [example] the MSE is very
> low but Spearman is 0.5, because a small error between 0.1 / 0.12 / 0.11 produces **a flip in the
> ranks**, which drags Spearman down to 0.5. So what I did: **I replaced the targets by their ranks.**
> If you have 0.1, 0.11, 0.2 you replace them with 1, 2, 3 — and even if the MSE is larger, the
> Spearman correlation is better. Those are two slightly extreme examples, but that's the idea that
> also helped me make a very good jump.
>
> "As for the algorithm: one small note — there are several **soft-ranking** variants, differentiable
> implementations of Spearman correlation; I experimented with PyTorch [and] XGBoost, but it's **not
> very stable** and it **overfits live after two or three iterations**. For the algorithm I tried
> several linear models — linear regression, LASSO — and tree models, and in the end **I decided to use
> mainly CatBoost, because it is very good with categorical data**, and **the most important feature is
> the COUNTRY feature** — Germany or France. Then I did a bit of **feature selection from the trained
> CatBoost**, and **trained it on the whole dataset**. That's all, thank you."

---

### Cross-verification of the load-bearing sentence

The one sentence everything hinges on was re-transcribed independently from the Collège de France
audio with `faster-whisper` (`small`, fr, beam 5). It agrees with the YouTube captions almost
word-for-word — both engines mangle the spoken initialism "ID" the same way (as *"idées" / "idies"*),
which is why it reads oddly in raw ASR:

```
586  juste en se basant sur cette création de feature et un petit changement sur le target, mais j'ai
590  aussi trouvé une correspondance entre les idées et les idies, genre le feature idies et les cluster.
598  Et donc j'ai fait un reclassement par idies et quand j'ai fait les plots, j'ai trouvé que je peux
603  reconstruire l'aspect[ temps], genre je trouve un time series. Donc ce que j'ai fait et j'ai
609  différencié les bases et après j'ai fait une différenciation[ par des moyennes mouvantes].
```

(Timestamps are Collège de France audio offsets, ≈37 s behind the YouTube video.)
Reading `idées`/`idies` → **ID**, the sentence is unambiguous: *"I also found a correspondence between
the IDs — the ID feature — and the clusters. So I re-sorted by ID, and when I plotted it I found I
could reconstruct the time aspect: I get a time series."*

---

## Structured summary of the method

| Stage | What he did |
|---|---|
| **Problem framing** | Explanatory (contemporaneous), not forecasting. Metric = Spearman on the test set. |
| **De-seasonalising (no leak)** | Multiple **clusterings of the days**, each fitted on a *different group of features* (weather group, consumption group, …). Each raw value is then **differenced against its own cluster's mean or median**. This is a data-driven stand-in for "subtract the July average" — it removes non-stationarity and implicitly recovers season. |
| **Target transform** | **Rank-transform the target** before fitting (0.1, 0.11, 0.2 → 1, 2, 3). Rationale: Spearman is rank-only, and least squares spends its budget on a handful of extreme days while flipping the ranks that actually score. |
| **Score without leak** | **0.32 Spearman, top 10–12**, from the two items above alone. |
| **Reverse-engineering the ordering** | Noticed a **correspondence between his cluster labels and the `ID` column**. **Re-sorting the rows by `ID` reconstructs the temporal ordering** — plotting the ID-sorted features shows clean time series. (See verification below: it is the row `ID`, not the anonymised `DAY_ID`.) |
| **Features from the ordering** | Once ordered: **first differences** of the series, and **differences against trailing moving averages of past values** (i.e. a value minus its rolling mean over previous days). These were "the main features that made the jump". |
| **Soft ranking** | Tried differentiable Spearman / soft-rank losses in PyTorch and XGBoost. **Rejected — unstable, overfits after 2–3 iterations.** |
| **Model** | Tried linear regression, LASSO, tree models. Settled on **CatBoost**, chosen because it handles **categorical features** well — and the single most important feature is **COUNTRY (FR vs DE)**. |
| **Feature selection / fitting** | Feature selection driven by the trained CatBoost's importances, then **retrained on the full dataset**. |
| **NOT mentioned** | No external data of any kind. No real gas/power price series, no weather archives, no calendar matching to real dates, no ensembling, no neighbouring-day target leakage, no volatility-regime modelling. Nothing beyond the competition CSVs. |

### What he explicitly did NOT claim

He never says he matched the anonymised days to **real-world calendar dates**. He says only that he
recovered the **relative ordering** (a time series), and used it to build differencing / rolling-mean
features. There is no mention of scraping real market data.

(For contrast: in the **CFM** stock-returns challenge presented later in the same video, at ~44:29–45:35,
**Amine Abdelkader** — who is also 3rd on this QRT electricity challenge — describes exactly the
external-data play: he *scraped four extra years of real market data and injected it*, and says he
could have gone further by identifying the actual years (2020–2021, Covid) and reconstructing the day
order from index add/drop events. That is a **different challenge**. Don't conflate the two.)

---

## Verification against the actual competition data

The talk's key claim was checked directly against `data/raw/X_train.csv` + `data/raw/X_test_final.csv`.
**It reproduces exactly, and the leak is in the `ID` column, not `DAY_ID`.**

- `DAY_ID` really is shuffled: sorted by `DAY_ID`, the day-level lag-1 autocorrelation of `DE_SOLAR`
  is **−0.001**, `FR_TEMP` **0.008**. Nothing.
- `ID` is not. Train ∪ test spans `ID` 0…2147 (2148 rows) over 1216 unique `DAY_ID`s, in two blocks:
  - **`ID` 0…1215** = exactly **one row per day, in chronological order** (932 `DE` + 284 `FR`-only days).
  - **`ID` 1216…2147** = the **`FR` twin** of the first 932 of those days, at exactly `ID + 1216`
    (verified: `ID − 1216` maps to the same `DAY_ID` for 100% of block-1 rows).
  - So *literally sorting the concatenated file by `ID`*, as he describes, already works — block 0 on
    its own is a complete chronological sweep of every day.
- Therefore **`t = ID % 1216` is a bijection onto the 1216 days** (verified: exactly 1 `DAY_ID` per `t`)
  **and it is the chronological index.** Day-level lag-1 autocorrelations under that ordering:

  | series | lag-1 autocorr sorted by `t = ID % 1216` |
  |---|---|
  | `FR_CONSUMPTION` | **0.951** |
  | `DE_SOLAR` | **0.882** |
  | `FR_SOLAR` | 0.873 |
  | `DE_CONSUMPTION` | 0.763 |
  | `FR_TEMP` | 0.754 |
  | `DE_WINDPOW` | 0.626 |
  | `GAS_RET` | 0.178 |
  | `COAL_RET` | −0.052 |

- The recovered series has a **clean annual cycle at ≈252 steps** — i.e. the days are **business
  days**. `DE_SOLAR` ACF peaks at lags **251–255** (0.66) and **495–507** (0.66), and troughs at
  lag 126 (−0.68). 1216 business days ≈ **4.8 years**.
- Train and test rows are **interleaved in time** (not a chronological holdout), so once ordered you
  have test days sandwiched between train days — which is what makes trailing/rolling features on the
  full concatenated panel legitimate *within the competition's own rules* and very powerful.
- Target autocorrelation is weak/negative (`FR` −0.03, `DE` −0.21 between consecutive *train* rows),
  so the gain is **not** from neighbouring days' targets — consistent with his account: the gain came
  from de-trending the *features* (differences and deviations from trailing moving averages).

Reproduce with:

```python
import pandas as pd, numpy as np
X  = pd.read_csv('data/raw/X_train.csv');  Xt = pd.read_csv('data/raw/X_test_final.csv')
A  = pd.concat([X, Xt], ignore_index=True)
A['t'] = A.ID % 1216                      # <- chronological business-day index
d = A.drop_duplicates('t').sort_values('t')
print(pd.Series(d.DE_SOLAR.values).autocorr(1))   # 0.882
```

Note: the repo's current `README.md` states *"`DAY_ID` is shuffled — no lags or ordering."* That is
true of `DAY_ID` and **false of `ID`**. This is the single highest-value finding here.

---

## Other sources chased (and what they yielded)

| Source | Result |
|---|---|
| Collège de France event page | Confirms date, speakers, and the three QRT winners. **No slides, no abstract beyond one line, no PDF.** Only the audio file. |
| `challengedata.ens.fr/winners/2023` | Confirms the podium. No writeups linked. |
| `challengedata.ens.fr/challenges/97` | Challenge description only. States "Dates have been anonymized, but all data corresponding to a specific day is consistent." |
| GitHub / GitLab search for Ben Amara | Nothing on this challenge. His only public repo (`gitlab.com/Dalyamara/research-internship`) is unrelated. |
| LinkedIn (`fr.linkedin.com/in/mohamed-ali-ben-amara`) | Blocked (HTTP 999); not retrievable without auth. |
| `challengedata.qube-rt.com` forum | Domain no longer resolves (DNS NXDOMAIN). |
| Other winners' writeups (`mb`, Amine Abdelkader) | None found for this challenge. |
| `github.com/97continuum/QRT---Electricity-Prices---2023` | A non-winner, 77th place / top 10%, LightGBM, 0.2749 Spearman. Not a winner's writeup. |

---

## Full transcript

The complete 54-minute French auto-caption transcript of the seminar (all challenge winners, not just
QRT) was retrieved and is reproducible with:

```
uv tool install youtube-transcript-api
youtube_transcript_api Xd4ESBwFn5s --languages fr --format json
```

The QRT portion is at **04:35–13:00**; the remainder covers OWKIN (PIK3CA breast-cancer mutation),
the prize list for all 2023 challenges (~24:00), and a panel with the top three of the **CFM**
US-equity end-of-session-returns challenge (~26:00–54:00).

---

## Appendix — raw verbatim ASR, 04:35–13:10 (unedited)

YouTube French auto-captions, exactly as returned. This is the raw material for the cleaned quotes above.

```
[04:35] Mohamed Ali Benamara tu peux venir
[04:40] maintenant voilà je laisse la parole
[04:46] merci bonjour à tous Je m'appelleardo
[04:49] peetti je travaille à kart et j'espère
[04:52] qu'il y a pas problème si je parle
[04:54] anglais c'est plus
[04:58] clair
[05:00] electricity prices it was not about
[05:03] predicting the price of the electricity
[05:04] in the future but explaining what was
[05:06] going on today with information that we
[05:08] have
[05:11] today as a benchmark for the score if
[05:14] you if you knew all this information you
[05:15] could get 100% prediction on the on the
[05:19] price you know if you know the prices of
[05:21] all the commodities of all the
[05:23] production in France all the production
[05:24] in Germany you could predict this
[05:26] perfectly so you know the best score
[05:27] would have been 100% so the SC that we
[05:30] got in the beginning of
[05:32] the we use data which is weather data
[05:37] data production in France from nuclear
[05:40] wind coal both in Germany also in
[05:43] Germany and you know it was a
[05:45] challenging problem because each of the
[05:48] countries have very different uh
[05:50] behaviors and uh we were very happy that
[05:54] there was so many people participating
[05:56] in in our challenge and something that
[06:00] that we saw you know like in the
[06:02] beginning scores were around 25%
[06:04] correlation to the score 27 and suddenly
[06:07] we saw a big jump in in scores and
[06:10] something that we noticed was that there
[06:12] there was a leak in the in the data in
[06:14] which you could uh we had anonymized the
[06:17] data in a way that you couldn't tell you
[06:19] know the dates and you know you could
[06:22] reproduce the you could sort The the the
[06:26] test set and that would give you some
[06:28] some information
[06:30] something that was very interesting
[06:32] after looking at many of the solutions
[06:34] from from students is that using this
[06:37] leak was not necessary to get a score
[06:39] one of the top scores you could really
[06:42] use arguments both fundamental using a
[06:45] real knowledge of the electricity market
[06:48] or understanding the seasonality of
[06:50] electricity through time to do this in a
[06:53] nonleak way so you know we saw really
[06:55] some
[06:56] brilliant
[06:57] solutions both using just machine
[07:00] learning using fundamental arguments
[07:03] using you know physics arguments for the
[07:05] seasonality and so so it was it was very
[07:08] interesting to see like all these very
[07:09] different solutions uh we hired a couple
[07:13] of people that participated in this
[07:14] challenge and gave us a a very
[07:16] interesting solution so you know for for
[07:18] us it was very good and I hope that the
[07:20] participants also enjoyed it so you know
[07:23] very interested to see in your
[07:24] resolution and congratulations and I
[07:26] just want to to thank Marie no een
[07:33] [Applaudissements]
[07:44] donc bonjour je suis Mohamed Ali benara
[07:47] je suis étudiant 3e année à l'in Paris
[07:49] et le M2 MVA et je vais vous présenter
[07:52] ma solution pour le
[07:57] challenge donc juste pour présenter le
[07:59] challenge un peu c'est le but est de
[08:03] d'expliquer ou de prédire euh les
[08:05] variations quotidiennes des prix de de
[08:07] l'électricité pour la France et
[08:09] l'Allemagne et pour comme variable
[08:12] explicative on a des données
[08:14] météorologiques quotidiennes dans la
[08:15] température pluie le vent et le prix des
[08:18] produits énergétiques et la consommation
[08:21] comme le photovoltaïque et le nucléaire
[08:23] et on a aussi les des des informations
[08:27] sur l'utilisation au quotidien de de
[08:29] l'électricité comm la consommation et
[08:31] les échanges et les les imports export
[08:33] entre les deux pays
[08:35] euh donc on veut prédire les variations
[08:38] quotidiennes mais on veut pas juste
[08:40] minimiser un erreur quadratique ou MSE
[08:44] ou MAE normal mais on veut maximiser la
[08:45] corrélation de
[08:48] spearmman
[08:50] donc les idées
[08:53] principales de de la solution sont juste
[08:56] dans la création des variables le
[08:58] feature engineering et un petit
[08:59] changement que j'ai fait sur le target
[09:01] alors pour la création des features on
[09:04] peut voir que les données de
[09:05] consommation et les données de
[09:06] météorologiques ne sont pas
[09:07] stationnaires al pour la température
[09:09] pendant l'été c'est pas vraiment la même
[09:12] distribution que la température pendant
[09:13] l'hiver et donc le premier challenge
[09:16] était de d'enlever cette
[09:18] nonstationnarité donc ce que j'ai fait
[09:20] et de d'essayer de créer des features
[09:22] basés sur des clusters je crée plusieurs
[09:24] clusters en se basant sur à chaque fois
[09:27] un groupe de featur et après je
[09:28] différencie par soit la moyenne ou la
[09:30] médiane de ce cluster pour avoir un peu
[09:32] de stationnarité et reconstruire un peu
[09:34] l'aspect temps genre par exemple si
[09:36] j'arrive à reconstruire
[09:38] euh un cluster qui qui groupe les dates
[09:42] de l'été et un autre pour l'hiver je
[09:44] peux différencier la variation de la
[09:46] cons je peux différencier la
[09:48] consommation ou la température dans ce
[09:50] dans ce jour-là par la moyenne en
[09:52] juillet ou pendant l'été et c'est un peu
[09:55] plus stationnaire et cette idée m'a j'ai
[09:59] j'ai eu une solution qui a 032 de score
[10:01] dans donc dans le top 10 ou le Top 12
[10:04] sans leak qui est juste en se basant sur
[10:06] cette création de feature et un petit
[10:07] changement sur le target mais j'ai aussi
[10:10] trouvé une correspondance entre les
[10:12] idées et les les ID genre le feature ID
[10:15] et les clusters et donc j'ai un j'ai
[10:18] fait un reclassement par ID et quand
[10:20] j'ai fait les plotes j'ai trouvé que je
[10:22] peux reconstruire l'aspect temp genre je
[10:24] trouve un time series donc ce que j'ai
[10:27] fait et j'ai différencié
[10:29] les les les bases et après j'ai j'ai
[10:32] fait un une différenciation par des
[10:34] moyennes mouvantes sur les valeurs
[10:35] passées et c'est ce sont les features
[10:38] principals qui qui m'ont aidé pour pour
[10:40] faire le
[10:41] Jump et après pour changer le target la
[10:44] corrélation de spearmman vérifie
[10:46] principalement les rangs des prédictions
[10:49] il faut avoir une relation monotone
[10:52] entre ce qu'on prédit est ce qu'on est
[10:54] ce qu'on a mais par contre la plupart
[10:57] des algorithmes optimise soit l'erreur
[10:59] quadratique ou absolu genre MSE ou Mae
[11:03] et et et même des petits changements sur
[11:05] dans l' MSE par exemple dans le premier
[11:08] exemple l' MSE est très il est très
[11:11] faible mais on a une corrélation de
[11:13] spirman de 0,5 car car le petite erreur
[11:16] entre 01 et 012 et 011 qui est prédite
[11:19] comme il faut on a un flip dans les dans
[11:22] les rangs et et ça et ça va induire une
[11:25] corrélation de spman de 05 donc ce que
[11:27] j'ai fait et j'ai remplacé j'ai remplacé
[11:30] les les targets par par leur rang si on
[11:35] a 01 011 et 02 on peut le remplacer par
[11:38] 1 2 3 et même si on a un un un MSE plus
[11:42] grand la corrélation de spirman elle est
[11:45] un peu plus
[11:46] importante elle est bonne ce sont deux
[11:49] exemples un peu extrêmes mais c'est
[11:50] l'idée qui m'a aussi aidé à faire un un
[11:53] très bon jump et pour l'algorithme euh
[11:57] j'ai essayé plusieurs modèles en se
[11:59] basant juste juste un un petit point sur
[12:02] sur ça aussi il y a plusieurs variations
[12:05] de soft ranking des des implémentations
[12:09] différenciaables de la corrélation de
[12:10] spearmman
[12:12] mais j'ai expérienté avec por chez XJ
[12:15] boost mais c'est pas très stable et
[12:17] c'est c'est un suapprentissage en direct
[12:20] après deux trois itérations
[12:22] et pour l'algorithme j'ai essayé
[12:24] plusieurs modèles linéaires genre de
[12:26] régression linéaire des lassau et des
[12:28] modèles sur les arbres et finalement
[12:30] j'ai décidé d'utiliser principalement le
[12:32] CAT boost car il est il est très bien
[12:34] avec les données catégoriques et la
[12:36] feature la plus importante et la feature
[12:38] de country gen Allemagne ou France et et
[12:42] après j'ai fait un petit feature un
[12:44] petit feature selection à partir du CAT
[12:47] boost que j'ai entraîné et je l'ai
[12:49] entraîné sur la toute la
[12:52] dataset c'est tout
[12:57] merci
[13:04] euh bonjour du coup on va passer à la
[13:07] 2è présentation d'un gagnant donc c'est
```
