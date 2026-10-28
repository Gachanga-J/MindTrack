# Phase 2: dependence-flag models (Member B)

## Setup
- **Target:** `dependence_flag` (TMD total 39 or more, "medium or high"). Train 976 students (439 flagged, 45.0%); test 244 (110 flagged, 45.1%). This is Member C's shared split in `data/processed/`, seed 42, the same students Member A used.
- **Inputs (41 columns):** age, semester, income level, family and phone-rule items, time since first phone, eight most-used-app flags, and the encoded background columns. These are the same inputs as Member A's F1, so the two targets can be compared directly.
- **Never used:** any `tmd_*` column (the target is built from them), `tmd_level`, `dependence_flag`, any `bdi_*` column or depression flag/level, `program`, `age_group`, `avg_daily_use_band`, ergonomics columns. The script stops with an error if a banned column slips in.
- **Models:** baseline (always "not dependent"), logistic regression, decision tree, random forest, gradient boosting. Settings chosen by 5-fold CV on train only (scored by AUC). Scaling is fitted inside each fold. The decision cut-off is picked from out-of-fold train predictions with one rule for every model. Test set used once.

## Test results (244 students)
| Model | Accuracy | Precision | Recall | F1 | Bal. acc. | Test AUC (95% range) | Train AUC |
|---|---|---|---|---|---|---|---|
| Baseline | 0.55 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| Logistic regression | 0.63 | 0.57 | 0.67 | 0.62 | 0.63 | 0.67 (0.60-0.73) | 0.69 |
| Decision tree | 0.59 | 0.54 | 0.65 | 0.59 | 0.60 | 0.66 (0.59-0.72) | 0.67 |
| Random forest | 0.61 | 0.55 | 0.77 | 0.64 | 0.63 | 0.70 (0.64-0.77) | 0.80 |
| Gradient boosting | 0.62 | 0.55 | 0.85 | 0.67 | 0.64 | 0.69 (0.63-0.76) | 0.67 |

Confusion counts on the test set (TN / FP / FN / TP): baseline 134 / 0 / 110 / 0; logistic 79 / 55 / 36 / 74; tree 73 / 61 / 39 / 71; forest 65 / 69 / 25 / 85; boosting 57 / 77 / 16 / 94.

Repeated 5x5 CV on train (more stable): logistic 0.64, tree 0.62, forest 0.64, boosting 0.63 (spread about plus or minus 0.03).

## What it means
1. **Accuracy is fair here, and the baseline is easy to beat on it.** The positive group is 45%, so "nobody is dependent" scores 55% accuracy but finds none of the 110 dependent students. The real models reach 59% to 63% accuracy and find 65% to 85% of them.
2. **The models are modestly better than chance, not strong.** Test AUC is 0.66 to 0.70 and balanced accuracy 0.60 to 0.64. When a model flags a student, about 54% to 57% of those flagged are truly dependent, against 45% in the whole group.
3. **The differences between models are within the noise.** Repeated CV puts logistic regression, random forest and gradient boosting within 0.02 of each other, which is less than the spread. The decision tree is lowest (0.62).
4. **Over-fitting is mild.** Random forest has the largest train-test gap (train 0.80, test 0.70). Logistic regression, the tree and boosting have gaps near zero, because their settings are kept simple (shallow trees, strong regularisation). Together with the low scores, this suggests there is little more to learn from these inputs.
5. **The signal is real but weak.** Shuffled labels give AUC 0.51 (highest 0.56) against 0.64 for the real labels.
6. **Same inputs, better score than depression.** Dependence reaches AUC about 0.64 to 0.70 against about 0.60 to 0.65 for the depression flag, which is plausible: phone habits and phone dependence are closer to each other than phone habits and mood.
7. **What the models lean on.** Across the tree-based models the top inputs are family activities, phone-related arguments at home, age, and in some models income level, discrimination over phone use, and whether the student uses a laptop or social media apps. Logistic regression also gives weight to department (Law, Engineering and Health above the base group), urban home and living arrangement. These are clues, not explanations.

## Caveats
- Dependence is a self-report score cut at 39, so the flag is a threshold on a scale, not a diagnosis.
- Phone-related arguments at home may partly be a result of heavy phone use, not only a cause. Say so when interpreting it, and check with Member C whether the app can collect it at all.
- With 110 flagged test students one test score is still noisy (95% range about plus or minus 0.07). Quote the repeated-CV figures as the main result.
- The 0.5 baseline AUC and 0.00 recall are the "always not dependent" rule, shown only for comparison.

## Files
`src/memberB_phase2_dependence.py`, `results/phase2_dependence_results.csv`, `..._confusion_matrices.csv`, `..._test_predictions.csv`, `..._test_auc_ci.csv`, `..._repeated_cv.csv`, `..._permutation_check.csv`, `..._top_features_{logistic,decision,random,gradient}.csv`, `models/phase2_dependence_models.joblib`.
