# Data cleaning: phone-dependence and depression files (Member B)

## What was cleaned
- **Dependence file:** Test of Mobile Dependence (TMD), 22 questions scored 0 to 4, total 0 to 88, 1,220 students.
- **Depression file:** Beck Depression Inventory, 21 questions scored 0 to 3, total 0 to 63, the same 1,220 students.
- Both files hold the same students in the same order as Member A's `sociodemo_clean.csv`, so they join one-to-one on `code`.
- Steps follow the seven cleaning steps in the ICS 3202 slides. Every decision is in `cleaning_log_memberB.csv` (19 rows).

## What was found and done
| Step | Finding | Decision |
|---|---|---|
| 1 Duplicates | 1 repeated row (TMD file) and 19 (Beck file) when the background columns are included; 455 students (37%) answered 0 on every Beck question | Kept all. Identical answers are genuine, and removing them would delete healthy students |
| 2 Irrelevant | Both files repeat 7 background columns already held by Member A (checked identical) | Dropped the repeats, kept `code` |
| 3 Incorrect | One student (code 164) has a supplied TMD total of 3, but the 22 answers add up to 64 (level should be High, not Absence). All Beck totals are correct | Recomputed both totals from the answers and rebuilt both level columns |
| 3 Incorrect | The instrument is labelled "BDI-2" and the phone test "TMDbrief", but the wording shows the **original** Beck inventory and the **full 22-item** TMD. One Beck item (N) is named wrongly in the dictionary | Reported under the correct names; see "Things to confirm" |
| 4 Structural | Merged header cells ("TMD    Q1", "DepressionA"), stray space in a column name | Renamed to `tmd_q1`..`tmd_q22`, `bdi_a`..`bdi_u`, `tmd_total`, `bdi_total` and so on |
| 5 Missing | 0 missing cells before and after merging | No imputation |
| 6 Outliers | TMD: 3 totals above the box-plot fence (83 to 88). Beck: 65 above (totals 21 to 48), because most students score near 0 | Kept all; they are real, bounded scores and the students we most want to detect |
| 7 Standardise | Questions are ordered whole numbers | Not scaled here: scaling must be fitted on the training part only (Member C) |

## Target columns
| Target | Rule | Students | Share |
|---|---|---|---|
| `depressed_flag` | Beck total 11 or more | 218 of 1,220 | 17.9% |
| `dependence_flag` | TMD total 39 or more (medium or high) | 549 of 1,220 | 45.0% |

The depression count matches Member A's (218). Level counts after rebuilding:
- **TMD:** Absence 442, Low 229, Medium 276, High 273.
- **Beck:** Normal 1,002, Mild 112, Borderline clinical 41, Moderate 43, Severe 21, Extreme 1.

Two sensitivity columns are stored but not used unless the team decides: `depressed_flag_excl_item_i` (total without the suicidal-ideation item; 10 students change group) and `depressed_flag_ge17` (borderline clinical or worse; 106 students, 8.7%).

## Things to confirm with the team and supervisor
- **The Beck cut-off.** This answers Member A's open question. The questionnaire is the original Beck inventory (work difficulty, weight loss in pounds, worry about physical problems, interest in sex), and the dictionary's bands are the standard bands for that form. The BDI-II bands (mild 14 to 19) do not apply. So 11 means "mild mood disturbance or worse" and 17 means "borderline clinical or worse". Pick one, say why, and keep `bdi_total` so it can be changed.
- **Item I (suicidal ideation).** It is part of the total used for the flag. The sensitivity column shows 10 students would change group without it. Decide whether to keep it, and report this item only in aggregate.
- **Citing the instrument.** If the report cites a Beck manual, cite the original inventory (Beck et al., 1961), not the BDI-II manual.
- **Dependence cut-off.** 39 or more is "medium or high" in the dictionary. It gives a 45% positive group, so this target is close to balanced.
- **Beck floor effect.** 37% of students scored 0, so higher depression levels are rare. Say this when explaining weak model scores.

## Hand-off to Member C
- Merge: `sociodemo_clean.csv` + `dependence_clean.csv` + `depression_clean.csv` on `code`. Result: 1,220 rows, 102 columns, 0 missing (tested).
- `targets_memberB.csv` holds `code`, both flags and both totals, if only the targets are needed.
- **Leakage rules** (also in `cleaned_data_dictionary_memberB.csv`):
  - Depression model: never use any `bdi_*` column. It may use the `tmd_*` questions (feature set F2).
  - Dependence model: never use any `tmd_*` column, `tmd_level` or `dependence_flag`. Depression columns are not planned as inputs.
  - Level columns are made from the totals, so they are for charts and the report only.
- Fit scalers and encoders on the training part only.

## Files
`dependence_clean.csv`, `depression_clean.csv`, `targets_memberB.csv`, `cleaning_log_memberB.csv`, `cleaned_data_dictionary_memberB.csv`, `score_distributions_memberB.png`, `memberB_cleaning.py` (set `DATA_DIR` and `SOCIO_CLEAN`, then run).
