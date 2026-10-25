"""
MindTrack - Member B pipeline (Colombian dataset)
Phase 1: clean the DEPENDENCE (TMD) and DEPRESSION (Beck) files, recompute totals,
         check the level labels and build the two target columns.

HOW TO RUN (Colab or local):
  1. Unzip the dataset and set DATA_DIR (folder that holds "3. Dataset_MobileDependence.xlsx" etc.).
  2. Put Member A's sociodemo_clean.csv somewhere and set SOCIO_CLEAN.
  3. Run:  python memberB_cleaning.py
Outputs are written to OUT_DIR.

Follows the seven cleaning steps in the ICS 3202 deck (duplicates, irrelevant, incorrect, structural,
missing, outliers, standardise/normalise) and Member A's conventions (snake_case names, `code` join key,
same cleaning-log columns).
Dataset: Digital Sedentary Behavior, Mobile Device Dependence, Musculoskeletal Symptoms, and Mental Health
among University Students (Mendeley Data, CC BY 4.0).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "/home/claude/c/Digital Sedentary Behavior, Mobile Device Dependen")
SOCIO_CLEAN = os.environ.get("SOCIO_CLEAN", "/mnt/user-data/uploads/sociodemo_clean.csv")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/user-data/outputs/memberB")
os.makedirs(OUT_DIR, exist_ok=True)

DEPRESSION_CUTOFF = 11      # team plan: Beck total >= 11 ("mild mood disturbance" or worse in the dictionary)
DEPENDENCE_CUTOFF = 39      # team plan: TMD total >= 39 ("medium" or "high" dependence in the dictionary)
ALT_DEPRESSION_CUTOFF = 17  # sensitivity only: "borderline clinical depression" or worse
log = []


def add_log(step, issue, affected, decision):
    log.append({"step": step, "issue": issue, "rows_or_columns_affected": affected,
                "decision": decision, "who": "Member B"})


# =============================================================== LOAD
tmd_raw = pd.read_excel(os.path.join(DATA_DIR, "3. Dataset_MobileDependence.xlsx"))
bdi_raw = pd.read_excel(os.path.join(DATA_DIR, "5. Dataset_Depression.xlsx"))
socio = pd.read_csv(SOCIO_CLEAN)
for d in (tmd_raw, bdi_raw):
    d.columns = [str(c).strip() for c in d.columns]
add_log("0 load", "Loaded the phone-dependence (TMD) and depression (Beck) files and Member A's sociodemo_clean.csv",
        f"TMD {tmd_raw.shape[0]}x{tmd_raw.shape[1]}, Beck {bdi_raw.shape[0]}x{bdi_raw.shape[1]}, "
        f"sociodemo_clean {socio.shape[0]}x{socio.shape[1]}", "Column names stripped of stray spaces")

# the first item column's header is merged with a label ("TMD    Q1", "DepressionA") - locate items by position
TMD_ITEMS_RAW = list(tmd_raw.columns[8:30])
BDI_ITEMS_RAW = list(bdi_raw.columns[8:29])
assert len(TMD_ITEMS_RAW) == 22 and tmd_raw.columns[30] == "sum" and tmd_raw.columns[31] == "Level_of_dependency"
assert len(BDI_ITEMS_RAW) == 21 and bdi_raw.columns[29] == "Sum"
BDI_LETTERS = list("abcdefghijklmnopqrstu")

# =============================================================== STEP 4 (structural): names, join key
tmd = pd.DataFrame({"code": tmd_raw["Code"]})
for i, c in enumerate(TMD_ITEMS_RAW, 1):
    tmd[f"tmd_q{i}"] = tmd_raw[c].astype(int)
tmd["tmd_total_supplied"] = tmd_raw["sum"]
tmd["tmd_level_supplied"] = tmd_raw["Level_of_dependency"].astype(str).str.strip()

bdi = pd.DataFrame({"code": bdi_raw["Code"]})
for letter, c in zip(BDI_LETTERS, BDI_ITEMS_RAW):
    bdi[f"bdi_{letter}"] = bdi_raw[c].astype(int)
bdi["bdi_total_supplied"] = bdi_raw["Sum"]
bdi["bdi_level_supplied"] = bdi_raw[bdi_raw.columns[30]].astype(str).str.strip()
add_log("4 structural",
        "Merged header cells ('TMD    Q1', 'DepressionA'), a stray space in 'Result_level_of_ depression', "
        "capitalised/unclear names (Q1-Q22, A-U, sum, Sum)", "TMD 32 cols, Beck 31 cols",
        "Renamed to tmd_q1..tmd_q22, tmd_total, tmd_level and bdi_a..bdi_u, bdi_total, bdi_level (snake_case, as Member A)")

# join key checks (against Member A's file)
assert tmd.code.is_unique and bdi.code.is_unique
assert (tmd.code.values == socio.code.values).all() and (bdi.code.values == socio.code.values).all()
add_log("4 structural", "Checked that code is unique and that both files hold the same 1,220 students in the same order "
        "as sociodemo_clean.csv", "3 files", "Merge on code is safe (one-to-one, no unmatched rows)")

# =============================================================== STEP 2 (irrelevant): repeated demographics
demo_cols = ["Program", "Department", "Semester", "Age", "Sex", "MS", "SES"]
label_maps = {
    "Department": ("department", {1: "Law", 2: "Economics", 3: "Engineering", 4: "Health"}),
    "Sex": ("sex", {1: "Male", 2: "Female"}),
    "MS": ("marital_status", {1: "Single", 2: "Married", 3: "Separated", 4: "Divorced", 5: "Widowed", 6: "Common-law"}),
    "SES": ("ses", {1: "Low-low", 2: "Low", 3: "Low-middle", 4: "Middle", 5: "High-middle", 6: "High"}),
}
same = {}
for name, raw in (("TMD", tmd_raw), ("Beck", bdi_raw)):
    ok = (raw["Age"].values == socio["age"].values).all() and (raw["Semester"].values == socio["semester"].values).all()
    for col, (sc, mp) in label_maps.items():
        ok = ok and (raw[col].map(mp).values == socio[sc].values).all()
    same[name] = bool(ok)
assert all(same.values()), same
add_log("2 irrelevant", "Both files repeat 7 background columns (program, department, semester, age, sex, marital status, "
        "SES) that already sit in Member A's file; checked that every value is identical", "14 columns (7 x 2 files)",
        "Dropped the repeated columns; kept code only as the link. Member C merges on code")

# =============================================================== STEP 1 (duplicates)
dup_tmd_full = int(tmd_raw.drop(columns="Code").duplicated().sum())
dup_bdi_full = int(bdi_raw.drop(columns="Code").duplicated().sum())
tmd_items = [f"tmd_q{i}" for i in range(1, 23)]
bdi_items = [f"bdi_{l}" for l in BDI_LETTERS]
dup_tmd_items = int(tmd[tmd_items].duplicated().sum())
dup_bdi_items = int(bdi[bdi_items].duplicated().sum())
n_zero_bdi = int((bdi[bdi_items].sum(axis=1) == 0).sum())
add_log("1 duplicates",
        f"Duplicates ignoring code (all columns incl. background): TMD file {dup_tmd_full}, Beck file {dup_bdi_full}. "
        f"Duplicates on the answers alone: TMD {dup_tmd_items}, Beck {dup_bdi_items}",
        f"{dup_tmd_full + dup_bdi_full} rows",
        f"Kept all. {n_zero_bdi} students answered 0 on every Beck item, so many identical rows are expected and genuine; "
        "removing them would delete healthy students and change the class balance")

# =============================================================== STEP 5 (missing) - before processing
n_miss = int(tmd_raw.isna().sum().sum() + bdi_raw.isna().sum().sum())
add_log("5 missing", f"Missing cells in the two raw files: {n_miss}", "none", "No imputation needed")

# =============================================================== STEP 3 (incorrect): ranges, totals, levels
tmd_rng = (int(tmd[tmd_items].min().min()), int(tmd[tmd_items].max().max()))
bdi_rng = (int(bdi[bdi_items].min().min()), int(bdi[bdi_items].max().max()))
assert tmd_rng == (0, 4) and bdi_rng == (0, 3)
add_log("3 incorrect", f"Item values outside the allowed range: TMD items allowed 0-4 (found {tmd_rng[0]}-{tmd_rng[1]}); "
        f"Beck items allowed 0-3 (found {bdi_rng[0]}-{bdi_rng[1]})", "none", "No recoding needed")

# --- TMD total
tmd["tmd_total"] = tmd[tmd_items].sum(axis=1)
bad_tmd = tmd.index[tmd.tmd_total != tmd.tmd_total_supplied].tolist()
detail = "; ".join(f"code {tmd.code[i]}: supplied {tmd.tmd_total_supplied[i]}, items add up to {tmd.tmd_total[i]}, "
                   f"supplied level '{tmd.tmd_level_supplied[i]}'" for i in bad_tmd)
add_log("3 incorrect", f"Supplied TMD total does not equal the sum of its 22 items for {len(bad_tmd)} student ({detail})",
        f"{len(bad_tmd)} row", "Recomputed tmd_total from the items for everyone. The one wrong supplied total looks like a typing error")
# --- Beck total
bdi["bdi_total"] = bdi[bdi_items].sum(axis=1)
bad_bdi = int((bdi.bdi_total != bdi.bdi_total_supplied).sum())
add_log("3 incorrect", f"Supplied Beck total vs sum of 21 items: {bad_bdi} mismatches", "none" if bad_bdi == 0 else str(bad_bdi),
        "Recomputed bdi_total from the items anyway (identical); maximum observed total is "
        f"{int(bdi.bdi_total.max())} of a possible 63")

# --- levels from the dictionary bands
def tmd_level(x):
    return "Absence" if x <= 29 else "Low" if x <= 38 else "Medium" if x <= 48 else "High"


def bdi_level(x):
    return ("Normal" if x <= 10 else "Mild" if x <= 16 else "Borderline clinical" if x <= 20 else
            "Moderate" if x <= 30 else "Severe" if x <= 40 else "Extreme")


tmd["tmd_level"] = tmd.tmd_total.map(tmd_level)
bdi["bdi_level"] = bdi.bdi_total.map(bdi_level)
tmd_lv_bad = int((tmd.tmd_level != tmd.tmd_level_supplied).sum())
bdi_lv_bad = int((bdi.bdi_level != bdi.bdi_level_supplied).sum())
add_log("3 incorrect", "Checked the supplied level labels against the dictionary bands (TMD: 0-29 Absence, 30-38 Low, "
        f"39-48 Medium, 49-88 High; Beck: 0-10, 11-16, 17-20, 21-30, 31-40, over 40). Mismatches: TMD {tmd_lv_bad}, Beck {bdi_lv_bad}",
        f"TMD {tmd_lv_bad}, Beck {bdi_lv_bad}",
        "Rebuilt both level columns from the recomputed totals. Beck 'Extreme' exists in the data "
        f"({int((bdi.bdi_level == 'Extreme').sum())} student, total {int(bdi.bdi_total.max())}) and its band is in the dictionary ('over 40')")

# --- instrument version (reading the questionnaire, not the sheet name)
add_log("3 incorrect",
        "The dictionary sheet and the questionnaire are both titled 'BDI-2', but the item wording (work difficulty, weight loss "
        "in pounds, worry about physical problems, interest in sex, body image) is the ORIGINAL 21-item Beck Depression Inventory, "
        "whose standard bands are exactly the ones in the dictionary (0-10 normal ... over 40 extreme)", "Beck file",
        "Report it as the Beck Depression Inventory (original form, bands as in the dictionary). The BDI-II bands (mild 14-19) do NOT apply, "
        "so a cut-off of 11 means 'mild mood disturbance or worse' and 17 means 'borderline clinical or worse'. Confirm with the supervisor")
add_log("3 incorrect",
        "Dictionary name for Beck item N is 'Feelings of Worthlessness', but the questionnaire statement for N is about body image "
        "('I am worried that I am looking old or unattractive...')", "1 item (bdi_n)",
        "Keep the column; describe it as body-image concern in the report. It is only part of the total, never a model input")
add_log("3 incorrect",
        "The dictionary sheet for the phone-dependence test is named 'TMDbrief', but it lists the full 22-item Test of Mobile Dependence "
        "(Q1-Q22, total 0-88)", "TMD file",
        "Report it as the 22-item Test of Mobile Dependence (TMD), not the brief version")

# --- answer-pattern checks
same_tmd = int((tmd[tmd_items].nunique(axis=1) == 1).sum())
add_log("3 incorrect", f"Students who gave the SAME answer to all 22 TMD questions: {same_tmd}; "
        f"students with all-zero Beck answers: {n_zero_bdi} ({n_zero_bdi / len(bdi):.0%})",
        f"{same_tmd} TMD rows, {n_zero_bdi} Beck rows",
        "Kept: all-zero Beck answers are plausible for students who feel well and the TMD pattern is rare. "
        "Mention the Beck 'floor effect' (many zeros) in the report")

# =============================================================== STEP 6 (outliers)
def fence(s):
    q1, q3 = s.quantile([0.25, 0.75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i


lo_t, hi_t = fence(tmd.tmd_total); lo_b, hi_b = fence(bdi.bdi_total)
n_out_t = int(((tmd.tmd_total < lo_t) | (tmd.tmd_total > hi_t)).sum())
n_out_b = int(((bdi.bdi_total < lo_b) | (bdi.bdi_total > hi_b)).sum())
add_log("6 outliers", f"Box-plot fences: TMD total {lo_t:.0f} to {hi_t:.0f} ({n_out_t} outside); Beck total {lo_b:.0f} to {hi_b:.0f} "
        f"({n_out_b} above, all above because most students score near 0)", f"TMD {n_out_t}, Beck {n_out_b}",
        "Kept all: both scores are bounded questionnaire totals (0-88, 0-63) and the high Beck scores are the students we most "
        "want to detect. No values removed or capped")

# =============================================================== TARGETS
tmd["dependence_flag"] = (tmd.tmd_total >= DEPENDENCE_CUTOFF).astype(int)
bdi["depressed_flag"] = (bdi.bdi_total >= DEPRESSION_CUTOFF).astype(int)
bdi["bdi_total_excl_item_i"] = bdi.bdi_total - bdi.bdi_i
bdi["depressed_flag_excl_item_i"] = (bdi.bdi_total_excl_item_i >= DEPRESSION_CUTOFF).astype(int)
bdi["depressed_flag_ge17"] = (bdi.bdi_total >= ALT_DEPRESSION_CUTOFF).astype(int)
n_dep, n_dependence = int(bdi.depressed_flag.sum()), int(tmd.dependence_flag.sum())
n_switch = int((bdi.depressed_flag != bdi.depressed_flag_excl_item_i).sum())
add_log("targets", f"depressed_flag = Beck total >= {DEPRESSION_CUTOFF}: {n_dep} of {len(bdi)} students ({n_dep / len(bdi):.1%}); "
        f"dependence_flag = TMD total >= {DEPENDENCE_CUTOFF}: {n_dependence} of {len(tmd)} ({n_dependence / len(tmd):.1%})",
        "2 new columns", "Both targets stored as 0/1. Totals are kept so the team can change a cut-off without re-cleaning")
add_log("targets", f"Sensitivity columns: depressed_flag_excl_item_i (Beck total without item I, suicidal ideation, same cut-off; "
        f"{n_switch} students change group) and depressed_flag_ge17 ({int(bdi.depressed_flag_ge17.sum())} students, "
        f"{bdi.depressed_flag_ge17.mean():.1%}, 'borderline clinical or worse')", "3 columns",
        "Not used unless the team decides; they answer the open questions about item I and the cut-off. Never use any Beck column as a model input")

# =============================================================== STEP 7 (standardise / normalise)
add_log("7 standardise", "All TMD questions are 0-4 ordered numbers, all Beck questions are 0-3 ordered numbers; both level columns are "
        "ordered text labels", "all columns",
        "No scaling or encoding done here on purpose: scaling must be fitted on the training part only (Member C). Level labels are "
        "for charts and the report, not model inputs, because they are made from the totals")

# =============================================================== missing check after processing + merge test
out_tmd = tmd[["code"] + tmd_items + ["tmd_total", "tmd_level", "dependence_flag"]].copy()
out_bdi = bdi[["code"] + bdi_items + ["bdi_total", "bdi_level", "depressed_flag", "bdi_total_excl_item_i",
                                       "depressed_flag_excl_item_i", "depressed_flag_ge17"]].copy()
merged = socio.merge(out_tmd, on="code", how="left", validate="1:1").merge(out_bdi, on="code", how="left", validate="1:1")
assert len(merged) == len(socio) == 1220 and merged.code.is_unique
assert int(out_tmd.isna().sum().sum() + out_bdi.isna().sum().sum()) == 0
add_log("5 missing", "Re-checked after cleaning and after merging with Member A's file (left join on code)",
        f"merged table {merged.shape[0]} rows x {merged.shape[1]} columns", "0 missing cells, 1,220 rows, no duplicated codes")

targets = pd.DataFrame({"code": tmd.code, "depressed_flag": bdi.depressed_flag, "dependence_flag": tmd.dependence_flag,
                        "bdi_total": bdi.bdi_total, "tmd_total": tmd.tmd_total})

# =============================================================== data dictionary (reads wording from the original dictionary)
dic = pd.ExcelFile(os.path.join(DATA_DIR, "1. Data_Dictionary.xlsx-QKG8M2.xlsx"))
tdesc = dic.parse("TMDbrief").set_index("Variable")["Description"].astype(str).str.replace("\n", " ").to_dict()
bdesc = dic.parse("BDI-2").set_index("Label")["Variable"].astype(str).to_dict()
rows = [{"file": "both", "column": "code", "type": "student id (join key)", "use_in_depression_model": "No",
         "use_in_dependence_model": "No", "note": "merge key; never a model input"}]
for i in range(1, 23):
    rows.append({"file": "dependence_clean.csv", "column": f"tmd_q{i}", "type": "ordered 0-4",
                 "use_in_depression_model": "Yes (feature set F2 only)", "use_in_dependence_model": "NO - target source",
                 "note": tdesc.get(f"Q{i}", "")})
rows += [{"file": "dependence_clean.csv", "column": "tmd_total", "type": "number 0-88", "use_in_depression_model": "Optional (F2)",
          "use_in_dependence_model": "NO - target source", "note": "recomputed from the 22 items (1 supplied total was wrong)"},
         {"file": "dependence_clean.csv", "column": "tmd_level", "type": "category (ordered)", "use_in_depression_model": "No",
          "use_in_dependence_model": "NO - target source", "note": "Absence / Low / Medium / High from the total; for charts and report only"},
         {"file": "dependence_clean.csv", "column": "dependence_flag", "type": "binary 0/1 TARGET", "use_in_depression_model": "No",
          "use_in_dependence_model": "TARGET", "note": f"1 = TMD total >= {DEPENDENCE_CUTOFF} (medium or high dependence)"}]
for l in BDI_LETTERS:
    rows.append({"file": "depression_clean.csv", "column": f"bdi_{l}", "type": "ordered 0-3",
                 "use_in_depression_model": "NO - target source", "use_in_dependence_model": "Not planned",
                 "note": (f"Beck item {l.upper()}: {bdesc.get(l.upper(), '')}" if l != "n" else
                          "Beck item N: body-image concern (the dictionary calls it 'Feelings of Worthlessness' but the questionnaire "
                          "statement is about looking unattractive)") +
                         (" (suicidal ideation; sensitive, aggregate results only)" if l == "i" else "")})
rows += [{"file": "depression_clean.csv", "column": "bdi_total", "type": "number 0-63", "use_in_depression_model": "NO - target source",
          "use_in_dependence_model": "Not planned", "note": "recomputed from the 21 items (matches the supplied total)"},
         {"file": "depression_clean.csv", "column": "bdi_level", "type": "category (ordered)", "use_in_depression_model": "NO - target source",
          "use_in_dependence_model": "Not planned", "note": "Normal / Mild / Borderline clinical / Moderate / Severe / Extreme; for charts and report only"},
         {"file": "depression_clean.csv", "column": "depressed_flag", "type": "binary 0/1 TARGET", "use_in_depression_model": "TARGET",
          "use_in_dependence_model": "No", "note": f"1 = Beck total >= {DEPRESSION_CUTOFF} (mild mood disturbance or worse)"},
         {"file": "depression_clean.csv", "column": "bdi_total_excl_item_i", "type": "number", "use_in_depression_model": "NO - target source",
          "use_in_dependence_model": "Not planned", "note": "sensitivity: total without the suicidal-ideation item"},
         {"file": "depression_clean.csv", "column": "depressed_flag_excl_item_i", "type": "binary 0/1 (sensitivity)", "use_in_depression_model": "Sensitivity target only",
          "use_in_dependence_model": "No", "note": f"1 = total without item I >= {DEPRESSION_CUTOFF}"},
         {"file": "depression_clean.csv", "column": "depressed_flag_ge17", "type": "binary 0/1 (sensitivity)", "use_in_depression_model": "Sensitivity target only",
          "use_in_dependence_model": "No", "note": f"1 = Beck total >= {ALT_DEPRESSION_CUTOFF} (borderline clinical or worse)"}]
dictionary = pd.DataFrame(rows)

# =============================================================== outputs
out_tmd.to_csv(f"{OUT_DIR}/dependence_clean.csv", index=False)
out_bdi.to_csv(f"{OUT_DIR}/depression_clean.csv", index=False)
targets.to_csv(f"{OUT_DIR}/targets_memberB.csv", index=False)
pd.DataFrame(log).to_csv(f"{OUT_DIR}/cleaning_log_memberB.csv", index=False)
dictionary.to_csv(f"{OUT_DIR}/cleaned_data_dictionary_memberB.csv", index=False)

# =============================================================== figure for the report (step 6 box-plots + level counts)
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), gridspec_kw={"height_ratios": [3, 1]})
for j, (s, name, cut, bins, col) in enumerate([(tmd.tmd_total, "TMD total (phone dependence, 0-88)", DEPENDENCE_CUTOFF, 22, "#2a7fb8"),
                                                (bdi.bdi_total, "Beck total (depression, 0-63)", DEPRESSION_CUTOFF, 25, "#c0504d")][:2]):
    ax[0, j].hist(s, bins=bins, color=col, alpha=0.85)
    ax[0, j].axvline(cut, color="black", ls="--", lw=1.2)
    ax[0, j].set_title(f"{name}\nflag cut-off = {cut}", fontsize=10)
    ax[0, j].set_ylabel("students")
    ax[1, j].boxplot(s, vert=False, widths=0.6)
    ax[1, j].set_yticks([]); ax[1, j].set_xlabel("score")
order_t = ["Absence", "Low", "Medium", "High"]
order_b = ["Normal", "Mild", "Borderline clinical", "Moderate", "Severe", "Extreme"]
ct = tmd.tmd_level.value_counts().reindex(order_t); cb = bdi.bdi_level.value_counts().reindex(order_b)
ax[0, 2].barh(order_b[::-1], cb[::-1].values, color="#c0504d", alpha=0.85, label="Beck level")
for k, v in enumerate(cb[::-1].values):
    ax[0, 2].text(v + 8, k, str(v), va="center", fontsize=8)
ax[0, 2].set_title("Beck level counts (after rebuilding from totals)", fontsize=10)
ax[1, 2].axis("off")
ax[1, 2].text(0, 0.5, "TMD levels: " + ", ".join(f"{k} {int(v)}" for k, v in ct.items()), fontsize=9, va="center")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/score_distributions_memberB.png", dpi=150)
plt.close()

print(f"Depression flag (Beck >= {DEPRESSION_CUTOFF}): {n_dep} of {len(bdi)} = {n_dep / len(bdi):.1%}")
print(f"Dependence flag (TMD >= {DEPENDENCE_CUTOFF}): {n_dependence} of {len(tmd)} = {n_dependence / len(tmd):.1%}")
print("Levels TMD:", ct.to_dict()); print("Levels Beck:", cb.to_dict())
print("Supplied-vs-rebuilt level mismatches: TMD", tmd_lv_bad, "| Beck", bdi_lv_bad)
print("Merged shape:", merged.shape, "| log rows:", len(log))
