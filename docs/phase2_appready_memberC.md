# Phase 2 (Member C): App-ready models and feature selection

**Goal.** Build models that use only the inputs the MindTrack app can collect from a student, and find out which of those inputs actually matter.

**Files.** Notebook: `notebooks/memberC_phase2_appready.ipynb` (explained step by step). Script: `src/memberC_phase2_appready.py` (same code). Selector: `src/memberC_selector.py`. Results: `results/phase2_appready_*.csv`. Charts: `docs/figures/appready_*.png`. Input list: `docs/app_inputs_memberC.csv`.

## 1. Inputs the app can collect (31)
* About the student: age, sex, department, semester, income level (`ses`), living arrangement, urban/rural.
* Phone: phone plan, network type, device type, first-phone timing, where the phone is mostly used.
* Apps: the eight "most used app" yes/no flags.

Excluded on purpose: family and phone-rule items, marital status, posture/ergonomics columns, `avg_daily_use_band` (coding looks broken), and every questionnaire answer (`tmd_*`, `bdi_*`) because the targets are built from them. A guard in the code stops the run if any excluded column appears on the input list.

## 2. Method (same recipe as Members A and B)
* Same shared split (976 train / 244 test, seed 42). Test set used once.
* Baseline + logistic regression, decision tree, random forest, gradient boosting; 5-fold CV grid search on train, scored by AUC; class weights balanced; the decision threshold is chosen on out-of-fold train predictions with the same rule as A and B.
* Two input versions: **all 31 inputs**, and **selected** (correlation filter at |r| >= 0.8, then top *k* of {8, 12, 18} by random-forest importance). Selection is done **inside every CV fold** (no leakage).
* Checks: 95% bootstrap range on test AUC, 5x5 repeated CV on train, shuffled-label check, permutation importance on the test set.
* Final model per target: highest repeated-CV AUC; anything within 0.01 counts as a tie; ties go to fewer inputs, then the simpler model.

## 3. Feature selection
* **Correlation with the targets is weak** (strongest 0.12 to 0.13, for age).
* **Near-duplicate inputs (|r| >= 0.8):** Wi-Fi vs cellular data (-0.99), home vs university use (-0.89), Health department vs smartphone (0.83), Health department vs first-phone timing (0.82). The filter dropped 3 inputs for depression and 4 for dependence (lists in `phase2_appready_selected_inputs.csv`).
* **Importance:** the selected models keep 8 to 18 inputs. Permutation importance is small and noisy for every input (age, department, place of use, residence area, income level and device type appear near the top). Treat as hints, not causes.

## 4. Results (5x5 repeated CV on train, AUC mean; 0.5 = coin flip)
| Target | Model | All 31 inputs | Selected | A/B larger list (F1) |
|---|---|---|---|---|
| Depression | Logistic regression | 0.593 | 0.590 | 0.603 |
| Depression | Decision tree | 0.611 | 0.610 | 0.596 |
| Depression | Random forest | 0.618 | 0.613 | 0.634 |
| Depression | Gradient boosting | 0.610 | 0.611 | 0.615 |
| Dependence | Logistic regression | 0.602 | 0.595 | 0.644 |
| Dependence | Decision tree | 0.550 | 0.548 | 0.617 |
| Dependence | Random forest | 0.598 | 0.584 | 0.643 |
| Dependence | Gradient boosting | 0.590 | 0.577 | 0.628 |

(Standard deviation across repeats: about 0.04 for depression, about 0.03 for dependence.)

**Final app-ready models:** depression flag = decision tree on 12 inputs (repeated-CV AUC 0.610); dependence flag = logistic regression on 18 inputs (0.595). For depression six models tie within 0.01; logistic regression on 18 inputs is an equally defensible choice (test AUC 0.63, train-test gap 0.01).

## 5. What it means
1. App-only inputs cost almost nothing for depression (change within -0.016 to +0.015) but cost 0.04 to 0.07 AUC for dependence, because the family and phone-rule items carried some of that signal.
2. Selection shortens the form to 8-18 inputs at a cost of about 0.015 AUC or less.
3. All models are weak screeners (AUC about 0.6). The shuffled-label check shows the signal is real (about 0.60 vs about 0.50) but small. For depression, 21 to 25 of 100 flagged students are truly flagged, against 18 of 100 by chance.
4. The 95% ranges on test AUC are about +/- 0.09 and overlap, so no model can be called better than another. Random forest and boosting overfit more (train-test gap 0.13 to 0.31) than logistic regression (0.01 to 0.05).
5. Limits: self-report screening flags, not diagnoses; one university sample (1,220 students); small test set (43 depression-flagged students).

## 6. Hand-over to Phase 3
`models/phase2_appready_final_models.joblib` holds, per target, the fitted pipeline, threshold and input list. Loading it needs `src/memberC_selector.py` on the Python path. `results/phase2_appready_test_predictions.csv` and `phase2_appready_confusion_matrices.csv` feed Member B's charts; `phase2_appready_results.csv`, `_repeated_cv.csv` and `_test_auc_ci.csv` feed Member A's metric tables.
