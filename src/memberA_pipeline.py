"""
MindTrack - Member A pipeline (Colombian dataset)
Phase 1: clean the background / phone-use file (49 columns)
Phase 2: depression-flag models (baseline + 4 models, two feature sets)
Phase 3: evaluation tables (train vs test, 5-fold CV, repeated CV)

HOW TO RUN (Colab or local):
  1. Put the unzipped dataset folder somewhere and set DATA_DIR below.
  2. Run:  python memberA_pipeline.py
Outputs are written to OUT_DIR.
Dataset: Digital Sedentary Behavior, Mobile Device Dependence, Musculoskeletal Symptoms,
and Mental Health among University Students (Mendeley Data, CC BY 4.0).
"""
import glob, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
SEED = 42                      # agree ONE seed with the team
DATA_DIR = "/home/claude/col/Digital Sedentary Behavior, Mobile Device Dependen"
OUT_DIR = "/mnt/user-data/outputs/memberA"
os.makedirs(OUT_DIR, exist_ok=True)

DEPRESSION_CUTOFF = 11         # team plan: Beck total >= 11 (dictionary's "mild mood disturbance")
log = []                       # cleaning log rows


def add_log(step, issue, affected, decision):
    log.append({"step": step, "issue": issue, "rows_or_columns_affected": affected,
                "decision": decision, "who": "Member A"})


# ------------------------------------------------------------------ PHASE 1
raw = pd.read_excel(os.path.join(DATA_DIR, "2. Dataset_Sociodemo.xlsx"))
raw.columns = [str(c).strip() for c in raw.columns]
add_log("0 load", "Loaded sociodemographic and phone-use file", f"{raw.shape[0]} rows x {raw.shape[1]} columns",
        "Column names stripped of stray spaces")

# --- Step 4 (structural): rename columns to clean snake_case names
rename = {
    "Code": "code", "Program": "program", "Department": "department", "Semester": "semester", "Age": "age",
    "Sex": "sex", "MS": "marital_status", "SES": "ses", "Living_Arrangement": "living_arrangement",
    "Family_Expression": "family_expression", "Family_Activities": "family_activities",
    "Mobile_Use_Rules": "mobile_use_rules", "Phone_Discrimination": "phone_discrimination",
    "Phone_Related_Arguments": "phone_related_arguments", "Residence_Area": "residence_area",
    "Subscription_Plan": "subscription_plan", "Network _Access": "network_access", "Areas_Use": "areas_use",
    "Time_First_Cellphone": "time_first_cellphone", "Type_MD": "type_md",
    "Using_Text Messaging_Apps": "app_messaging", "Using_Games_Apps": "app_games",
    "Using_social media_Apps": "app_social_media", "Using_Music_Apps": "app_music",
    "Using_Videos_Apps": "app_video", "Using_Ttime-management_Apps": "app_time_management",
    "Using_Camara_Apps": "app_camera", "Using_Mail_Apps": "app_mail",
    "Average_Time_Use": "avg_daily_use_band", "Hand_Dominance": "hand_dominance",
    "Body Position_Standing": "pos_standing", "Body Position_Walking": "pos_walking",
    "Body Position_Sitting on the Floor": "pos_sit_floor", "Body Position_Sitting Supported": "pos_sit_supported",
    "Body Position_Sitting Unsupported": "pos_sit_unsupported", "Body Position_Lying on One's Side": "pos_lying_side",
    "Body Position_Lying Face Down": "pos_lying_face_down", "Body Position_Lying Face Up": "pos_lying_face_up",
    "Usage time_Standing": "time_standing", "Usage time_Walking": "time_walking",
    "Usage time_Sitting on the Floor": "time_sit_floor", "Usage time_Sitting Supported": "time_sit_supported",
    "Usage time_Sitting Unsupported": "time_sit_unsupported", "Usage time_Lying on One's Side": "time_lying_side",
    "Usage time_Lying Face Down": "time_lying_face_down", "Usage time_Lying Face Up": "time_lying_face_up",
    "Usual_Phone_Locationn": "usual_phone_location", "Primary_input_finger": "primary_input_finger",
    "Frequent_touch_gesture": "frequent_touch_gesture",
}
missing_cols = [c for c in raw.columns if c not in rename]
assert not missing_cols, f"Unmapped columns: {missing_cols}"
df = raw.rename(columns=rename)
add_log("4 structural", "Typos / stray spaces in column names (e.g. 'Usual_Phone_Locationn', 'Network _Access', "
        "'Using_Camara_Apps', 'Using_Ttime-management_Apps')", "13 column names", "Renamed all 49 columns to clean snake_case")

# --- Step 1: duplicates
dup_all = int(df.duplicated().sum())
dup_nocode = int(df.drop(columns="code").duplicated().sum())
add_log("1 duplicates", f"Full-row duplicates: {dup_all}. Duplicates ignoring Code: {dup_nocode}",
        f"{dup_nocode} rows",
        "Nothing to remove in this file" if dup_nocode == 0 else
        "Kept: students can genuinely give identical answers on mostly yes/no and 1-4 items")
assert df.code.is_unique

# --- Step 5: missing values
n_missing = int(df.isna().sum().sum())
add_log("5 missing", f"Missing cells: {n_missing}", "none", "No imputation needed")

# --- Step 3: incorrect data - check every code is inside the range the dictionary allows
expected = {
    "program": range(1, 20), "department": range(1, 5), "semester": range(1, 11), "age": range(16, 61),
    "sex": [1, 2], "marital_status": range(1, 7), "ses": range(1, 7), "living_arrangement": range(1, 7),
    "family_expression": range(1, 5), "family_activities": range(1, 5), "mobile_use_rules": range(1, 5),
    "phone_discrimination": range(1, 5), "phone_related_arguments": range(1, 5), "residence_area": [1, 2],
    "subscription_plan": [1, 2], "network_access": range(1, 4), "areas_use": range(1, 5),
    "time_first_cellphone": range(1, 6), "type_md": range(1, 5), "avg_daily_use_band": range(1, 5),
    "hand_dominance": [1, 2], "usual_phone_location": range(1, 4), "primary_input_finger": range(1, 6),
    "frequent_touch_gesture": range(1, 6),
}
for c in [c for c in df.columns if c.startswith(("app_", "pos_"))]:
    expected[c] = [1, 2]
for c in [c for c in df.columns if c.startswith("time_") and c != "time_first_cellphone"]:
    expected[c] = range(1, 5)
bad = {c: int((~df[c].isin(list(r))).sum()) for c, r in expected.items() if (~df[c].isin(list(r))).sum() > 0}
add_log("3 incorrect", f"Values outside the dictionary's allowed codes: {bad if bad else 'none'}", "none" if not bad else str(bad),
        "No recoding needed" if not bad else "Review")

# daily use band
band = df.avg_daily_use_band.value_counts().sort_index().to_dict()
add_log("3 incorrect", f"avg_daily_use_band looks implausible: counts by code {band}. Only 1 student is in '1-3 hours' "
        "while 56% are in 'more than 6 hours' (code 4).", "1 column",
        "Kept in the clean file but EXCLUDED from the main models; used only in a sensitivity check. Ask the dataset authors")
add_log("3 incorrect", "Data dictionary has typos in the 'Usage time_*' rows (the lead number runs 2,3,4...8 instead of 1). "
        "The questionnaire and the data both use 1=<1h, 2=1-2:59h, 3=3-6h, 4=>6h.", "8 columns",
        "Used the questionnaire coding (1-4) for all eight usage-time columns")
add_log("3 incorrect", "The 'apps used' questions ask which apps the student uses MOST OFTEN (tick several), not which apps "
        "they use at all. So a 0 for social media means 'not among my most used', not 'never uses'.", "8 columns",
        "Kept as 0/1 'most-used app' flags and describe them this way in the report")

# --- Step 6: outliers
q1, q3 = df.age.quantile([0.25, 0.75]); iqr = q3 - q1
hi = q3 + 1.5 * iqr
n_hi = int((df.age > hi).sum()); n_minor = int((df.age < 18).sum()); n_over30 = int((df.age > 30).sum())
add_log("6 outliers", f"Age range {df.age.min()}-{df.age.max()}; {n_hi} students above the box-plot fence ({hi:.0f}); "
        f"{n_over30} are older than 30; {n_minor} are under 18", f"{n_hi} rows",
        "Kept: they are real students. Added an age_group column; ask supervisor whether under-18s need separate handling")
df["age_group"] = pd.cut(df.age, [0, 17, 20, 25, 35, 100], labels=["16-17", "18-20", "21-25", "26-35", "36+"])

# --- decode categories with the dictionary
labels = {
    "department": {1: "Law", 2: "Economics", 3: "Engineering", 4: "Health"},
    "program": {1: "Law", 2: "Administration", 3: "Economics", 4: "Finance", 5: "Marketing", 6: "Accounting",
                7: "Bioengineering", 8: "Electrical Engineering", 9: "Industrial Engineering", 10: "Energy Engineering",
                11: "Systems Engineering", 12: "Medicine", 13: "Physical Therapy", 14: "Nursing", 15: "Dentistry",
                16: "Respiratory Therapy", 17: "Surgical Technology", 18: "Speech-Language Pathology", 19: "Psychology"},
    "sex": {1: "Male", 2: "Female"},
    "marital_status": {1: "Single", 2: "Married", 3: "Separated", 4: "Divorced", 5: "Widowed", 6: "Common-law"},
    "ses": {1: "Low-low", 2: "Low", 3: "Low-middle", 4: "Middle", 5: "High-middle", 6: "High"},
    "living_arrangement": {1: "Both parents", 2: "Mother/Father only", 3: "Relatives", 4: "Roommate",
                           5: "Partner", 6: "Alone"},
    "residence_area": {1: "Urban", 2: "Rural"}, "subscription_plan": {1: "Prepaid", 2: "Postpaid"},
    "network_access": {1: "Cellular data", 2: "Wi-Fi", 3: "Both"},
    "areas_use": {1: "Home", 2: "University", 3: "Shopping centres", 4: "All"},
    "time_first_cellphone": {1: "<6 months", 2: "6-12 months", 3: "1-3 years", 4: "3-6 years", 5: ">6 years"},
    "type_md": {1: "Cell phone", 2: "Smartphone", 3: "Tablet", 4: "Laptop"},
    "hand_dominance": {1: "Right", 2: "Left"},
    "usual_phone_location": {1: "Carried / on lap", 2: "On a table", 3: "In the air / not resting"},
    "primary_input_finger": {1: "One thumb", 2: "Both thumbs", 3: "One index finger", 4: "Both index fingers",
                             5: "Multiple fingers"},
    "frequent_touch_gesture": {1: "Dragging", 2: "Scrolling/Swiping", 3: "Tapping", 4: "Pinching/Zooming",
                               5: "Other/Multiple"},
}
clean = df.copy()
for c, m in labels.items():
    clean[c] = df[c].map(m)
for c in [c for c in df.columns if c.startswith(("app_", "pos_"))]:
    clean[c] = (df[c] == 2).astype(int)     # 1=No, 2=Yes -> 0/1
add_log("4 structural", "Numbers used as codes for categories (e.g. sex 1/2, program 1-19) and 1=No/2=Yes for yes/no items",
        "Most columns", "Replaced category codes with labels; yes/no items turned into 0/1; ordered 1-4 items left as numbers")
add_log("2 irrelevant", "Hand dominance, finger, touch gesture, phone location, 8 body-position and 8 position-time columns "
        "belong to the muscle-and-joint part of the original study", "19 columns",
        "Kept in sociodemo_clean.csv for completeness but NOT used as model inputs (model_use = No in the dictionary)")
add_log("2 irrelevant", "program has 19 categories and overlaps with department", "1 column",
        "Use department (4 groups) in the models; keep program in the clean file")

# --- data dictionary for the clean file
type_map = {c: "binary 0/1" for c in clean.columns if c.startswith(("app_", "pos_"))}
type_map.update({c: "ordered 1-4" for c in ["family_expression", "family_activities", "mobile_use_rules",
                                            "phone_discrimination", "phone_related_arguments", "avg_daily_use_band"]})
type_map.update({c: "ordered 1-4" for c in clean.columns if c.startswith("time_") and c != "time_first_cellphone"})
type_map.update({c: "number" for c in ["age", "semester"]})
type_map["code"] = "student id (join key)"
rows = []
excluded = {"code", "program", "avg_daily_use_band", "hand_dominance", "usual_phone_location", "primary_input_finger",
            "frequent_touch_gesture", "age_group"}
excluded |= {c for c in clean.columns if c.startswith(("pos_", "time_")) and c != "time_first_cellphone"}
for c in clean.columns:
    rows.append({"column": c, "type": type_map.get(c, "category (label)"),
                 "model_use": "No" if c in excluded else "Yes",
                 "note": ("UNRELIABLE coding; sensitivity check only" if c == "avg_daily_use_band" else
                          "most-used-app flag" if c.startswith("app_") else
                          "ergonomics item, not mental-health relevant" if c in excluded and c not in ("code", "program", "age_group") else "")})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/cleaned_data_dictionary.csv", index=False)
clean.to_csv(f"{OUT_DIR}/sociodemo_clean.csv", index=False)

# ------------------------------------------------------------------ PHASE 2
bdi = pd.read_excel(os.path.join(DATA_DIR, "5. Dataset_Depression.xlsx")); bdi.columns = [str(c).strip() for c in bdi.columns]
tmd = pd.read_excel(os.path.join(DATA_DIR, "3. Dataset_MobileDependence.xlsx")); tmd.columns = [str(c).strip() for c in tmd.columns]
assert (bdi.Code.values == raw.Code.values).all() and (tmd.Code.values == raw.Code.values).all()
item_cols = [bdi.columns[8]] + list("BCDEFGHIJKLMNOPQRSTU")
assert (bdi[item_cols].sum(axis=1) == bdi.Sum).all(), "BDI total does not match its items"
y = (bdi.Sum >= DEPRESSION_CUTOFF).astype(int).values
tmd_items = [tmd.columns[8]] + [f"Q{i}" for i in range(2, 23)]
T = tmd[tmd_items].copy(); T.columns = [f"tmd_q{i}" for i in range(1, 23)]
data = pd.concat([clean.reset_index(drop=True), T.reset_index(drop=True)], axis=1)
data["depressed_flag"] = y
n_pos = int(y.sum())
print(f"Depression flag (BDI total >= {DEPRESSION_CUTOFF}): {n_pos} of {len(y)} = {y.mean():.1%}")

nominal = ["sex", "marital_status", "ses", "department", "living_arrangement", "residence_area", "subscription_plan",
           "network_access", "areas_use", "time_first_cellphone", "type_md"]
numeric = ["age", "semester", "family_expression", "family_activities", "mobile_use_rules", "phone_discrimination",
           "phone_related_arguments"] + [c for c in clean.columns if c.startswith("app_")]
F1 = nominal + numeric
F2 = F1 + list(T.columns)
S1 = F1 + ["avg_daily_use_band"]                       # sensitivity: add the unreliable daily-use column
sets = {"F1 phone & background": F1, "F2 + dependence questions": F2}

idx = np.arange(len(data))
tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
pd.DataFrame({"code": data.code.values[tr_idx], "split": "train"}).pipe(
    lambda a: pd.concat([a, pd.DataFrame({"code": data.code.values[te_idx], "split": "test"})])
).to_csv(f"{OUT_DIR}/split_memberA_seed42.csv", index=False)
print(f"Train {len(tr_idx)} (positives {int(y[tr_idx].sum())}) | Test {len(te_idx)} (positives {int(y[te_idx].sum())})")


def make_pre(cols):
    nom = [c for c in cols if c in nominal]
    num = [c for c in cols if c not in nominal]
    return ColumnTransformer([("nom", OneHotEncoder(handle_unknown="ignore"), nom),
                              ("num", StandardScaler(), num)])


def model_zoo(cols):
    pre = make_pre(cols)
    return {
        "Baseline (most common)": (Pipeline([("pre", pre), ("m", DummyClassifier(strategy="most_frequent"))]), {}),
        "Logistic regression": (Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
                                {"m__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}),
        "Decision tree": (Pipeline([("pre", pre), ("m", DecisionTreeClassifier(class_weight="balanced", random_state=SEED))]),
                          {"m__max_depth": [2, 3, 4, 5], "m__min_samples_leaf": [10, 20, 40]}),
        "Random forest": (Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                                              random_state=SEED, n_jobs=-1))]),
                          {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [5, 10, 20]}),
        "Gradient boosting": (Pipeline([("pre", pre), ("m", GradientBoostingClassifier(random_state=SEED))]),
                              {"m__max_depth": [1, 2, 3], "m__learning_rate": [0.03, 0.1], "m__n_estimators": [50, 150]}),
    }


def metrics(yt, pred, prob):
    return {"accuracy": accuracy_score(yt, pred), "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred), "f1": f1_score(yt, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "auc": roc_auc_score(yt, prob) if len(set(yt)) > 1 else np.nan}


cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, fitted = [], {}
for sname, cols in sets.items():
    Xtr, Xte = data.iloc[tr_idx][cols], data.iloc[te_idx][cols]
    ytr, yte = y[tr_idx], y[te_idx]
    for mname, (pipe, grid) in model_zoo(cols).items():
        if mname == "Baseline (most common)":
            best = pipe.fit(Xtr, ytr); cv_auc, cv_sd, params = 0.5, 0.0, {}
        else:
            fit_kw = {}
            if mname == "Gradient boosting":
                fit_kw = {"m__sample_weight": compute_sample_weight("balanced", ytr)}
            gs = GridSearchCV(pipe, grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
            gs.fit(Xtr, ytr, **fit_kw)
            best, params = gs.best_estimator_, gs.best_params_
            cv_auc = gs.best_score_
            cv_sd = gs.cv_results_["std_test_score"][gs.best_index_]
        ptr = best.predict_proba(Xtr)[:, 1]; pte = best.predict_proba(Xte)[:, 1]
        if mname == "Gradient boosting":                    # threshold at the training positive rate, as balanced models do at 0.5
            thr = float(np.quantile(ptr, 1 - ytr.mean()))
        else:
            thr = 0.5
        mtr = metrics(ytr, (ptr >= thr).astype(int), ptr); mte = metrics(yte, (pte >= thr).astype(int), pte)
        row = {"feature_set": sname, "model": mname, "cv5_auc_mean(train)": cv_auc, "cv5_auc_sd": cv_sd,
               "train_auc": mtr["auc"], "train_balanced_acc": mtr["balanced_accuracy"]}
        row.update({f"test_{k}": v for k, v in mte.items()})
        row["auc_gap_train_minus_test"] = mtr["auc"] - mte["auc"]
        row["best_params"] = json.dumps({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()})
        results.append(row); fitted[(sname, mname)] = (best, thr)

res = pd.DataFrame(results)
res.round(3).to_csv(f"{OUT_DIR}/depression_results_memberA.csv", index=False)

# bootstrap confidence interval for the test AUC of each model (test set is small)
rng = np.random.default_rng(SEED)
ci_rows = []
for (sname, mname), (best, thr) in fitted.items():
    if mname.startswith("Baseline"): continue
    Xte = data.iloc[te_idx][sets[sname]]; yte = y[te_idx]; p = best.predict_proba(Xte)[:, 1]
    boots = []
    for _ in range(1000):
        b = rng.integers(0, len(yte), len(yte))
        if len(set(yte[b])) > 1: boots.append(roc_auc_score(yte[b], p[b]))
    ci_rows.append({"feature_set": sname, "model": mname, "test_auc": roc_auc_score(yte, p),
                    "auc_95ci_low": np.percentile(boots, 2.5), "auc_95ci_high": np.percentile(boots, 97.5)})
pd.DataFrame(ci_rows).round(3).to_csv(f"{OUT_DIR}/test_auc_confidence_intervals.csv", index=False)

# stability check: repeated 5x10 CV on ALL students with fixed, sensible settings (no tuning on the folds)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
fixed = lambda cols: {
    "Baseline (most common)": Pipeline([("pre", make_pre(cols)), ("m", DummyClassifier(strategy="most_frequent"))]),
    "Logistic regression": Pipeline([("pre", make_pre(cols)), ("m", LogisticRegression(C=0.03, max_iter=3000, class_weight="balanced"))]),
    "Decision tree": Pipeline([("pre", make_pre(cols)), ("m", DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=SEED))]),
    "Random forest": Pipeline([("pre", make_pre(cols)), ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
    "Gradient boosting": Pipeline([("pre", make_pre(cols)), ("m", GradientBoostingClassifier(max_depth=2, learning_rate=0.05, n_estimators=100, random_state=SEED))]),
}
rep_rows = []
all_sets = dict(sets); all_sets["F1 + daily-use column (sensitivity)"] = S1
for sname, cols in all_sets.items():
    for mname, pipe in fixed(cols).items():
        if mname.startswith("Baseline"):
            rep_rows.append({"feature_set": sname, "model": mname, "repeated_cv_auc_mean": 0.5, "repeated_cv_auc_sd": 0.0}); continue
        s = cross_val_score(pipe, data[cols], y, cv=rcv, scoring="roc_auc", n_jobs=-1)
        rep_rows.append({"feature_set": sname, "model": mname, "repeated_cv_auc_mean": s.mean(), "repeated_cv_auc_sd": s.std()})
rep = pd.DataFrame(rep_rows); rep.round(3).to_csv(f"{OUT_DIR}/depression_repeated_cv_memberA.csv", index=False)

# permutation check: does the best set beat shuffled labels?
perm = []
shuf = np.random.default_rng(SEED)
pipe = fixed(F2)["Logistic regression"]
real = cross_val_score(pipe, data[F2], y, cv=rcv, scoring="roc_auc", n_jobs=-1).mean()
for _ in range(20):
    ys = shuf.permutation(y)
    perm.append(cross_val_score(pipe, data[F2], ys, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                                scoring="roc_auc", n_jobs=-1).mean())
pd.DataFrame([{"check": "F2 logistic regression repeated-CV AUC", "real_labels": real,
               "shuffled_labels_mean": np.mean(perm), "shuffled_labels_max": np.max(perm)}]).round(3).to_csv(
    f"{OUT_DIR}/permutation_check_memberA.csv", index=False)

# what drives the models (top logistic-regression coefficients, F2) - for Member B's charts
best_lr, _ = fitted[("F2 + dependence questions", "Logistic regression")]
names = best_lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(best_lr.named_steps["m"].coef_[0], index=names).sort_values(key=abs, ascending=False)
coef.head(15).round(3).rename("standardised_coefficient").to_csv(f"{OUT_DIR}/top_features_logreg_F2.csv")

pd.DataFrame(log).to_csv(f"{OUT_DIR}/cleaning_log_memberA.csv", index=False)
print("Done. Files in", OUT_DIR)
