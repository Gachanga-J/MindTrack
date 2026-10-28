# Phase 2 improvements and sensitivity checks (Member A)

All figures: repeated stratified 5-fold cross-validation (5 repeats = 25 folds) on all 1,220 students, fixed model settings (nothing tuned on the folds). Scalers, encoders and the input-selection step are fitted inside each training fold. No Beck answer is ever an input. The target is described as "mild mood disturbance or higher" (Beck total 11+), not a clinical diagnosis. Positive group: 218 of 1,220 (17.9%).

## 1. Headline result (replaces single-test-set scores)
| Inputs | Model | AUC (± sd) | PR-AUC | Precision at 70% recall |
|---|---|---|---|---|
| none | Baseline | 0.50 | 0.18 | 0.18 |
| F1 | Logistic regression | 0.61 ± 0.04 | 0.26 | 0.23 |
| F1 | Random forest | 0.64 ± 0.04 | 0.27 | 0.25 |
| F1 | Gradient boosting | 0.64 ± 0.06 | 0.31 | 0.25 |
| F1 | Decision tree | 0.59 ± 0.05 | 0.23 | 0.22 |
| F2 | Logistic regression | 0.59 ± 0.04 | 0.25 | 0.22 |
| F2 | Random forest | 0.64 ± 0.05 | 0.29 | 0.24 |
| F2 | Gradient boosting | 0.64 ± 0.05 | 0.29 | 0.24 |
| F2 | Decision tree | 0.57 ± 0.04 | 0.21 | 0.20 |

Reading it: to catch 70% of the flagged students, the best models are right about 1 time in 4 when they raise a flag (baseline: 18%). That is a modest gain, enough for a "check in" prompt, not a screening result.

## 2. What the added ideas showed
- **Fewer inputs (elastic net, forest on top 15):** no gain. Elastic net matches logistic regression (0.61 / 0.59); forest on 15 inputs is slightly worse than on all of them (0.63 / 0.61 vs 0.64 / 0.64). The signal is spread thinly across many inputs, not concentrated in a few.
- **Compact phone-dependence score (F3: total score, 1 column):** the best idea. Forest AUC 0.65 and logistic regression 0.61, both at or above F2 (22 columns), and the simplest for the app to collect.
- **PCA of the 22 questions (F4):** same as F3 (0.65 forest, 0.61 logistic). A published subscale split was not available in the data files, so I did not invent one.
- **Ensemble (logistic + forest + boosting):** 0.64 on F1, 0.62 on F2. Not better than its best member.
- **Calibration:** the calibrated ensemble was unstable (AUC 0.48 on F1, with some folds near 0.4) because the signal is too weak for the sigmoid step to fit reliably. Treat this as a negative result. Gradient boosting (unweighted) already has the best Brier score (0.14, against 0.147 for always predicting the group rate), so its probabilities are usable as they are.
- **Predicting the Beck score instead of the flag:** the scores correlate with the true totals at Spearman 0.32 to 0.38. Used as a ranking for the 11+ flag they give AUC 0.59 to 0.64, the same as the direct flag models. So the weak result is not caused by the cut-off.

## 3. Sensitivity checks (change in AUC versus the main flag, averaged over 6 model/input combinations)
| Check | Students | Flagged | Mean AUC change | Range |
|---|---|---|---|---|
| S1: Beck 17+ | 1,220 | 106 (8.7%) | -0.060 | -0.081 to -0.044 |
| S2: item I removed from total | 1,220 | 208 (17.0%) | -0.002 | -0.014 to +0.009 |
| S3: under-18s removed | 1,205 | 215 (17.8%) | +0.002 | -0.005 to +0.008 |

- 10 students change group when item I is removed. 15 students are under 18 (ages 16 to 17); 3 are flagged.
- Conclusions hold under S2 and S3. Under S1 the models get weaker (AUC 0.54 to 0.59) because there are few positives; say this openly.

## 4. What to claim in the report
- Phone use and background carry **limited but real** information about mild mood disturbance (AUC about 0.64, shuffled labels 0.50).
- Random forest and gradient boosting are best; logistic regression is the most honest and interpretable. Differences between the top models are inside the noise (sd about 0.04 to 0.06).
- Phone-dependence information helps only slightly, and as one total score it is as good as 22 separate questions.
- A higher score would need inputs that overlap with the questionnaire itself (leakage), so it is not a goal.

## 5. Still to confirm
- Ethics cover for the under-18s (ask the supervisor). Results are the same with or without them.
- The population and sampling statement: quote it from the Mendeley record and attribute it to the dataset authors. I could not open the record, so I have not verified it.
- Under-18s and item I must be reported only in aggregate.
