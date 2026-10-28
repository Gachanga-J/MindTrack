"""
MindTrack - Member A, Phase 2: depression-flag models
Baseline + logistic regression, decision tree, random forest, gradient boosting.

Uses the team's SHARED split from Member C (data/processed/train.csv and test.csv),
so Member A, B and C all evaluate on the same students.

Run from the project root:   python src/memberA_phase2_depression.py
Outputs go to results/ and models/.

Rules followed
  * Target: depressed_flag (Beck total >= 11). Change TARGET to try the other flags.
  * Leakage: no bdi_* column, no depression flag/level, no dependence_flag is ever an input.
  * Scaling and encoding are fitted INSIDE each cross-validation fold (Pipeline), on the
    training part only. That is why unscaled train.csv is used, not train_scaled.csv.
  * Test set is touched once, after all settings are chosen with 5-fold CV on train.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "depressed_flag"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models")
os.makedirs(RES, exist_ok=True); os.makedirs(MOD, exist_ok=True)

train = pd.read_csv(os.path.join(PROC, "train.csv"))
test = pd.read_csv(os.path.join(PROC, "test.csv"))
assert not set(train.code) & set(test.code), "train and test overlap"
ytr, yte = train[TARGET].values, test[TARGET].values
print(f"Train {len(train)} ({ytr.sum()} flagged, {ytr.mean():.1%}) | Test {len(test)} ({yte.sum()} flagged, {yte.mean():.1%})")

# ---------------------------------------------------------------- feature sets
ordinal_numeric = ["age", "semester", "ses", "family_expression", "family_activities", "mobile_use_rules",
                   "phone_discrimination", "phone_related_arguments", "time_first_cellphone"]
apps = [c for c in train.columns if c.startswith("app_")]
dummies = [c for c in train.columns if c.startswith(("department_", "sex_", "marital_status_", "living_arrangement_",
                                                       "residence_area_", "subscription_plan_", "network_access_",
                                                       "areas_use_", "type_md_"))]
tmd_items = [f"tmd_q{i}" for i in range(1, 23)]
F1 = ordinal_numeric + apps + dummies          # phone use & background
F2 = F1 + tmd_items                            # + the 22 phone-dependence questions
sets = {"F1 phone & background": F1, "F2 + dependence questions": F2}
# excluded on purpose: bdi_* (target source), program (overlaps department), age_group (copy of age),
# avg_daily_use_band (unreliable coding), ergonomics columns, tmd_total/tmd_level (sums of the items)
banned = [c for c in F2 if c.startswith("bdi_") or "depress" in c or c == "dependence_flag"]
assert not banned, banned


def pre(cols):
    scale = [c for c in cols if c in ordinal_numeric + tmd_items]
    return ColumnTransformer([("scale", StandardScaler(), scale)], remainder="passthrough")  # 0/1 dummies untouched


def zoo(cols):
    p = pre(cols)
    mk = lambda m: Pipeline([("pre", p), ("m", m)])
    return {
        "Baseline (most common)": (mk(DummyClassifier(strategy="most_frequent")), {}),
        "Logistic regression": (mk(LogisticRegression(max_iter=3000, class_weight="balanced")),
                                {"m__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}),
        "Decision tree": (mk(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
                          {"m__max_depth": [2, 3, 4, 5], "m__min_samples_leaf": [10, 20, 40]}),
        "Random forest": (mk(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
                          {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [5, 10, 20]}),
        "Gradient boosting": (mk(GradientBoostingClassifier(random_state=SEED)),
                              {"m__max_depth": [1, 2, 3], "m__learning_rate": [0.03, 0.1], "m__n_estimators": [50, 150]}),
    }


def scores(y, pred, prob):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, prob)}


def pick_threshold(y, prob):
    """Same rule for every model: the cut-off that maximises balanced accuracy on out-of-fold TRAIN predictions."""
    fpr, tpr, thr = roc_curve(y, prob)
    return float(thr[np.argmax(tpr - fpr)])


cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, preds, fitted = [], {}, {}
for sname, cols in sets.items():
    Xtr, Xte = train[cols], test[cols]
    for mname, (pipe, grid) in zoo(cols).items():
        gb = mname == "Gradient boosting"
        fit_kw = {"m__sample_weight": compute_sample_weight("balanced", ytr)} if gb else {}
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv5, scoring="roc_auc", n_jobs=1)
            gs.fit(Xtr, ytr, **fit_kw)
            best, params, cv_auc = gs.best_estimator_, gs.best_params_, gs.best_score_
            cv_sd = gs.cv_results_["std_test_score"][gs.best_index_]
            oof = cross_val_predict(best, Xtr, ytr, cv=cv5, method="predict_proba", params=fit_kw)[:, 1]
            thr = pick_threshold(ytr, oof)
        else:
            best = pipe.fit(Xtr, ytr); params, cv_auc, cv_sd, thr = {}, 0.5, 0.0, 0.5
        ptr, pte = best.predict_proba(Xtr)[:, 1], best.predict_proba(Xte)[:, 1]
        if not grid:                                   # baseline: predicts "not depressed" for everyone
            mtr = scores(ytr, np.zeros_like(ytr), ptr); mte = scores(yte, np.zeros_like(yte), pte)
        else:
            mtr = scores(ytr, (ptr >= thr).astype(int), ptr); mte = scores(yte, (pte >= thr).astype(int), pte)
        r = {"feature_set": sname, "model": mname, "cv5_auc_mean(train)": cv_auc, "cv5_auc_sd": cv_sd, "threshold": thr,
             "train_auc": mtr["auc"], "train_balanced_acc": mtr["balanced_accuracy"]}
        r.update({f"test_{k}": v for k, v in mte.items()})
        r["auc_gap_train_minus_test"] = mtr["auc"] - mte["auc"]
        r["best_params"] = json.dumps({k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()})
        rows.append(r); fitted[(sname, mname)] = (best, thr, cols)
        preds[(sname, mname)] = pte
        print(f"{sname:28s} {mname:24s} cv5 AUC {cv_auc:.3f} | test AUC {mte['auc']:.3f} | recall {mte['recall']:.2f} | gap {r['auc_gap_train_minus_test']:.2f}", flush=True)

res = pd.DataFrame(rows)
res.round(3).to_csv(os.path.join(RES, "phase2_depression_results.csv"), index=False)

# test-set predictions (for Member B's confusion matrices / ROC curves)
pr = test[["code"]].copy(); pr["y_true"] = yte
for (s, m), p in preds.items():
    pr[f"{s.split()[0]}|{m}|prob"] = p
    pr[f"{s.split()[0]}|{m}|pred"] = (p >= fitted[(s, m)][1]).astype(int) if not m.startswith("Baseline") else 0
pr.round(4).to_csv(os.path.join(RES, "phase2_depression_test_predictions.csv"), index=False)

# bootstrap 95% interval on test AUC (only 43 flagged students in test)
rng = np.random.default_rng(SEED); ci = []
for (s, m), p in preds.items():
    if m.startswith("Baseline"): continue
    b = [roc_auc_score(yte[i], p[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(1000)) if len(set(yte[i])) > 1]
    ci.append({"feature_set": s, "model": m, "test_auc": roc_auc_score(yte, p),
               "auc_95ci_low": np.percentile(b, 2.5), "auc_95ci_high": np.percentile(b, 97.5)})
pd.DataFrame(ci).round(3).to_csv(os.path.join(RES, "phase2_depression_test_auc_ci.csv"), index=False)

# stability: 5x5 repeated CV on the TRAINING part with each model's chosen settings (test set stays untouched)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED); rep = []
for (s, m), (best, thr, cols) in fitted.items():
    if m.startswith("Baseline"):
        rep.append({"feature_set": s, "model": m, "repeated_cv_auc_mean": 0.5, "repeated_cv_auc_sd": 0.0}); continue
    fk = {"params": {"m__sample_weight": compute_sample_weight("balanced", ytr)}} if m == "Gradient boosting" else {}
    sc = cross_val_score(best, train[cols], ytr, cv=rcv, scoring="roc_auc", **fk)
    rep.append({"feature_set": s, "model": m, "repeated_cv_auc_mean": sc.mean(), "repeated_cv_auc_sd": sc.std()})
pd.DataFrame(rep).round(3).to_csv(os.path.join(RES, "phase2_depression_repeated_cv.csv"), index=False)

# permutation check on the best-tuned logistic regression (F2): real labels vs shuffled labels
pipe = fitted[("F2 + dependence questions", "Logistic regression")][0]
real = cross_val_score(pipe, train[F2], ytr, cv=rcv, scoring="roc_auc").mean()
shuf = np.random.default_rng(SEED)
perm = [cross_val_score(pipe, train[F2], shuf.permutation(ytr), cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                        scoring="roc_auc").mean() for _ in range(20)]
pd.DataFrame([{"check": "F2 logistic regression", "real_labels_auc": real, "shuffled_mean": np.mean(perm),
               "shuffled_max": np.max(perm)}]).round(3).to_csv(os.path.join(RES, "phase2_depression_permutation_check.csv"), index=False)

# top logistic-regression coefficients (F1 and F2) for Member B's charts
for s in sets:
    best = fitted[(s, "Logistic regression")][0]
    names = [n.split("__", 1)[1] for n in best.named_steps["pre"].get_feature_names_out()]
    coef = pd.Series(best.named_steps["m"].coef_[0], index=names).sort_values(key=abs, ascending=False)
    coef.head(15).round(3).rename("standardised_coefficient").to_csv(os.path.join(RES, f"phase2_depression_logreg_top_features_{s.split()[0]}.csv"))

joblib.dump({f"{s}|{m}": v[0] for (s, m), v in fitted.items()}, os.path.join(MOD, "phase2_depression_models.joblib"))
print("Done. Results in", RES)
