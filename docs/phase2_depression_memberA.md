# Phase 2: depression-flag models (Member A)

## Setup
- **Target:** `depressed_flag` (Beck total 11 or more). Train 976 students (175 flagged, 17.9%); test 244 (43 flagged, 17.6%). This is Member C's shared split in `data/processed/`, seed 42.
- **Inputs F1:** age, semester, income level, family and phone-rule items, time since first phone, eight most-used-app flags, and the encoded background columns.
- **Inputs F2:** F1 plus the 22 phone-dependence questions.
- **Never used:** any `bdi_*` column, depression flags/levels, `dependence_flag`, `program`, `avg_daily_use_band`, ergonomics columns.
- **Models:** baseline (always "not depressed"), logistic regression, decision tree, random forest, gradient boosting. Settings chosen by 5-fold CV on train only (scored by AUC). Scaling is fitted inside each fold. The decision cut-off is picked from out-of-fold train predictions with one rule for every model. Test set used once.

## Test results (244 students)
| Inputs | Model | Recall | Precision | F1 | Bal. acc. | Test AUC (95% range) | Train AUC |
|---|---|---|---|---|---|---|---|
| none | Baseline | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| F1 | Logistic regression | 0.84 | 0.21 | 0.34 | 0.58 | 0.60 (0.50-0.68) | 0.69 |
| F1 | Decision tree | 0.49 | 0.16 | 0.24 | 0.46 | 0.52 (0.41-0.61) | 0.70 |
| F1 | Random forest | 0.74 | 0.22 | 0.34 | 0.58 | 0.65 (0.56-0.74) | 0.93 |
| F1 | Gradient boosting | 0.67 | 0.21 | 0.32 | 0.57 | 0.61 (0.52-0.70) | 0.72 |
| F2 | Logistic regression | 0.33 | 0.27 | 0.30 | 0.57 | 0.60 (0.51-0.69) | 0.69 |
| F2 | Decision tree | 0.56 | 0.17 | 0.26 | 0.48 | 0.49 (0.40-0.57) | 0.75 |
| F2 | Random forest | 0.77 | 0.22 | 0.35 | 0.60 | 0.63 (0.55-0.72) | 1.00 |
| F2 | Gradient boosting | 0.37 | 0.18 | 0.24 | 0.51 | 0.55 (0.46-0.64) | 1.00 |

Repeated 5x5 CV on train (more stable): F1 0.60 / 0.60 / 0.63 / 0.62 and F2 0.60 / 0.61 / 0.65 / 0.63 for logistic / tree / forest / boosting.

## What it means
1. The baseline's 82% accuracy is misleading: it finds none of the 43 flagged students.
2. All models are only slightly better than chance (AUC about 0.5 to 0.65). Shuffled labels give 0.50, real labels 0.60, so the signal is weak but real.
3. Random forest is the most consistent (best repeated-CV AUC, 0.63 to 0.65), but it over-fits (train AUC 0.93 to 1.00). Logistic regression generalises most honestly (gap about 0.09).
4. The decision tree is no better than chance on the test set; its CV score does not hold up.
5. Adding the dependence questions (F2) did not clearly help.
6. Accuracy looks low for most models because the cut-off was chosen to balance sensitivity and specificity. Report recall, balanced accuracy and AUC, not accuracy.

## Caveats
- With 43 flagged test students, one test score is noisy. Quote the repeated-CV figures as the main result.
- Beck cut-off (11 vs 17) and inclusion of item I are still open team decisions; change `TARGET` in the script to try `depressed_flag_ge17` or `depressed_flag_excl_item_i`.
- This is exploratory screening of a self-report scale, not a diagnosis.
