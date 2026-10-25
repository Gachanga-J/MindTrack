# Evaluation results: depression flag (Member A)

## What was tested
- **Target:** a student is flagged when their Beck depression total is 11 or more (218 of 1,220 students, 17.9%). This is the cut-off the team agreed from the dataset's own dictionary. See "Things to confirm" below.
- **Inputs, set F1 (phone and background):** age, sex, marital status, family income level, semester, department, living arrangement, family communication and activities, household phone rules, phone-related arguments and discrimination, urban or rural home, phone plan, network type, where the phone is mostly used, when the first phone was got, device type, and eight "most-used app" flags.
- **Inputs, set F2:** everything in F1 plus the 22 phone-dependence questions.
- **Never used as inputs:** any depression question, the anxiety file, and the daily-use column (its coding looks broken). The daily-use column was only tried in a side check.
- **Split:** 80% of students for learning (976, of whom 174 flagged) and 20% for the final test (244, of whom 44 flagged), with the same flagged share in both. Random seed 42.
- **Models:** a "guess the most common answer" baseline, logistic regression, decision tree, random forest and gradient boosting. Settings were chosen with 5-fold cross-validation on the learning part only. The test part was used once.

## Results on the final test set (244 students)

| Inputs | Model | Accuracy | Precision | Recall | F1 | Balanced accuracy | AUC | Train AUC |
|---|---|---|---|---|---|---|---|---|
| none | Baseline | 0.82 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| F1 | Logistic regression | 0.64 | 0.25 | 0.52 | 0.34 | 0.59 | 0.62 | 0.71 |
| F1 | Decision tree | 0.53 | 0.22 | 0.66 | 0.33 | 0.58 | 0.56 | 0.69 |
| F1 | Random forest | 0.75 | 0.30 | 0.30 | 0.30 | 0.57 | 0.65 | 0.93 |
| F1 | Gradient boosting | 0.78 | 0.36 | 0.27 | 0.31 | 0.58 | 0.69 | 0.82 |
| F2 | Logistic regression | 0.65 | 0.24 | 0.45 | 0.32 | 0.57 | 0.61 | 0.74 |
| F2 | Decision tree | 0.57 | 0.21 | 0.48 | 0.29 | 0.54 | 0.51 | 0.73 |
| F2 | Random forest | 0.81 | 0.25 | 0.02 | 0.04 | 0.50 | 0.65 | 1.00 |
| F2 | Gradient boosting | 0.76 | 0.27 | 0.20 | 0.23 | 0.54 | 0.61 | 0.95 |

AUC of 0.5 means a coin flip and 1.0 means perfect. Because the test set has only 44 flagged students, each test AUC has a wide margin: the 95% range is roughly plus or minus 0.08 (for example gradient boosting on F1: 0.69, range 0.60 to 0.77).

## What the numbers mean
1. **The baseline's 82% accuracy is misleading.** It scores well by calling nobody depressed, so it finds none of the 44 flagged students (recall 0). Balanced accuracy, AUC and recall are the fairer measures.
2. **The models are only slightly better than chance.** Test AUC is between 0.51 and 0.69 and balanced accuracy between 0.50 and 0.59. When a model flags someone, only about 21% to 36% of those flagged are truly depressed, against 18% in the whole group. That is a small gain.
3. **Over-fitting is clear in the random forest and gradient boosting.** The random forest on F2 scores a perfect 1.00 on the learning data but 0.65 on the test data, so it memorised the learning students. Logistic regression and the shallow tree have a smaller gap but also lower scores, which suggests there is little to learn.
4. **The differences between models are within the noise.** Repeating 5-fold cross-validation ten times on all 1,220 students gives AUC of about 0.57 to 0.65 across models (the two decision trees are lowest), with a spread of about plus or minus 0.04. Gradient boosting on F1 (0.646) and random forest on F2 (0.643) are the highest, but they are not reliably better than logistic regression (0.61 and 0.60).
5. **Adding the phone-dependence questions did not help.** F2 scored no higher than F1 in the repeated cross-validation. Adding the doubtful daily-use column changed nothing either.
6. **The signal is weak but real.** A check with randomly shuffled depression labels gave AUC of 0.50 on average (highest 0.55), against 0.60 with the real labels for logistic regression on F2. So the models are picking up something, though not much.
7. **What the models lean on is mostly background, not phone habits.** The strongest logistic-regression inputs include device type, department, where the phone is used, who the student lives with, urban or rural home and income level. The phone-dependence questions carry little weight. Treat these as clues, not explanations.

## Things to confirm with the team and supervisor
- **The cut-off.** The dictionary's Beck bands (0 to 10 "normal ups and downs", 11 to 16 "mild mood disturbance", and so on) look like the older Beck scoring. My understanding is that the standard BDI-II bands put "mild" at 14 to 19, so a cut-off of 11 may be too low. Please check this against the Beck manual or the instrument in the data file, and re-run with the agreed number.
- **The suicide question.** Beck item I (suicidal ideation) is part of the total used for the flag. The team should decide whether to leave it in. The models never use it as an input.
- **Small test set.** With 44 flagged students in the test part, any single test score is unstable. Quote the repeated cross-validation figures as the main result.
- **Under-18s.** 15 students are under 18 (aged 16 or 17). Ask the supervisor if they need special handling.
- **Wording.** This is exploratory screening of a self-report questionnaire, not a diagnosis. Say so wherever results are reported.

## Files for the rest of the team
- `sociodemo_clean.csv` and `cleaned_data_dictionary.csv` (Member C merges on `code`)
- `cleaning_log_memberA.csv`
- `depression_results_memberA.csv`, `test_auc_confidence_intervals.csv`, `depression_repeated_cv_memberA.csv`, `permutation_check_memberA.csv`, `top_features_logreg_F2.csv` (Member B can chart these)
- `split_memberA_seed42.csv` (replace with Member C's shared split when it exists)
- `memberA_pipeline.py` (set `DATA_DIR` and run)
