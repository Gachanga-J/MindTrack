"""
MindTrack - Member B, Phase 2: dependence-flag models
Baseline + logistic regression, decision tree, random forest, gradient boosting.

Mirrors Member A's depression script (memberA_phase2_depression.py) so the two targets are
directly comparable. Uses the team's SHARED split from Member C
(data/processed/train.csv and test.csv).

Run from the project root:   python src/memberB_phase2_dependence.py
Outputs go to results/ and models/.

Rules followed
  * Target: dependence_flag (TMD total >= 39, "medium or high"). 45% positive, so it is close to
    balanced and plain accuracy is meaningful (unlike the depression flag).
  * Leakage: NO tmd_* column (questions, total, level) and no dependence_flag is ever an input,
    because the target is built from them. No bdi_* column or depression flag/level either
    (the app cannot collect a Beck questionnaire, and the team plan keeps depression out of inputs).
  * Scaling is fitted INSIDE each cross-validation fold (Pipeline), on the training part only.
  * Settings are chosen with 5-fold CV on train only (scored by AUC). The decision cut-off is picked
    from out-of-fold TRAIN predictions with one rule for every model. The test set is touched once.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "dependence_flag"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results"); MOD = os.path.join(ROOT, "models")
os.makedirs(RES, exist_ok=True); os.makedirs(MOD, exist_ok=True)

train = pd.read_csv(os.path.join(PROC, "train.csv"))
test = pd.read_csv(os.path.join(PROC, "test.csv"))
assert not set(train.code) & set(test.code), "train and test overlap"
ytr, yte = train[TARGET].values, test[TARGET].values
print(f"Train {len(train)} ({ytr.sum()} flagged, {ytr.mean():.1%}) | Test {len(test)} ({yte.sum()} flagged, {yte.mean():.1%})")

# ---------------------------------------------------------------- input set
# Same phone-use and background inputs as Member A's F1, so the two targets use identical inputs.
ordinal_numeric = ["age", "semester", "ses", "family_expression", "family_activities", "mobile_use_rules",
                   "phone_discrimination", "phone_related_arguments", "time_first_cellphone"]
apps = [c for c in train.columns if c.startswith("app_")]
dummies = [c for c in train.columns if c.startswith(("department_", "sex_", "marital_status_", "living_arrangement_",
                                                       "residence_area_", "subscription_plan_", "network_access_",
                                                       "areas_use_", "type_md_"))]
F1 = ordinal_numeric + apps + dummies
# excluded on purpose: tmd_* (target source), bdi_* / depression (not an app input), program (overlaps
# department), age_group (copy of age), avg_daily_use_band (unreliable coding), ergonomics columns
banned = [c for c in F1 if c.startswith(("tmd_", "bdi_")) or "depress" in c or "dependence" in c]
assert not banned, banned
print(f"{len(F1)} input columns")


def pre():
    scale = [c for c in F1 if c in ordinal_numeric]
    return ColumnTransformer([("scale", StandardScaler(), scale)], remainder="passthrough")  # 0/1 columns untouched


def zoo():
    mk = lambda m: Pipeline([("pre", pre()), ("m", m)])
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
            "balanced_accuracy": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob)}


def pick_threshold(y, prob):
    """Same rule for every model: the cut-off that maximises balanced accuracy on out-of-fold TRAIN predictions."""
    fpr, tpr, thr = roc_curve(y, prob)
    return float(thr[np.argmax(tpr - fpr)])


cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, preds, fitted, cms = [], {}, {}, []
Xtr, Xte = train[F1], test[F1]
for mname, (pipe, grid) in zoo().items():
    if grid:
        gs = GridSearchCV(pipe, grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
        gs.fit(Xtr, ytr)
        best, params, cv_auc = gs.best_estimator_, gs.best_params_, gs.best_score_
        cv_sd = gs.cv_results_["std_test_score"][gs.best_index_]
        oof = cross_val_predict(best, Xtr, ytr, cv=cv5, method="predict_proba")[:, 1]
        thr = pick_threshold(ytr, oof)
    else:
        best = pipe.fit(Xtr, ytr); params, cv_auc, cv_sd, thr = {}, 0.5, 0.0, 0.5
    ptr, pte = best.predict_proba(Xtr)[:, 1], best.predict_proba(Xte)[:, 1]
    if not grid:                                   # baseline: predicts "not dependent" for everyone
        ctr, cte = np.zeros_like(ytr), np.zeros_like(yte)
    else:
        ctr, cte = (ptr >= thr).astype(int), (pte >= thr).astype(int)
    mtr, mte = scores(ytr, ctr, ptr), scores(yte, cte, pte)
    r = {"model": mname, "cv5_auc_mean(train)": cv_auc, "cv5_auc_sd": cv_sd, "threshold": thr,
         "train_auc": mtr["auc"], "train_balanced_acc": mtr["balanced_accuracy"]}
    r.update({f"test_{k}": v for k, v in mte.items()})
    r["auc_gap_train_minus_test"] = mtr["auc"] - mte["auc"]
    r["best_params"] = json.dumps({k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()})
    rows.append(r); fitted[mname] = (best, thr); preds[mname] = (pte, cte)
    tn, fp, fn, tp = confusion_matrix(yte, cte, labels=[0, 1]).ravel()
    cms.append({"model": mname, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    print(f"{mname:24s} cv5 AUC {cv_auc:.3f} | test AUC {mte['auc']:.3f} | acc {mte['accuracy']:.2f} | "
          f"recall {mte['recall']:.2f} | prec {mte['precision']:.2f} | gap {r['auc_gap_train_minus_test']:.2f}", flush=True)

res = pd.DataFrame(rows)
res.round(3).to_csv(os.path.join(RES, "phase2_dependence_results.csv"), index=False)
pd.DataFrame(cms).to_csv(os.path.join(RES, "phase2_dependence_confusion_matrices.csv"), index=False)

# test-set predictions (for confusion matrices / ROC curves in the evaluation phase)
pr = test[["code"]].copy(); pr["y_true"] = yte
for m, (p, c) in preds.items():
    pr[f"{m}|prob"] = p; pr[f"{m}|pred"] = c
pr.round(4).to_csv(os.path.join(RES, "phase2_dependence_test_predictions.csv"), index=False)

# bootstrap 95% interval on test AUC
rng = np.random.default_rng(SEED); ci = []
for m, (p, _) in preds.items():
    if m.startswith("Baseline"): continue
    b = [roc_auc_score(yte[i], p[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(1000)) if len(set(yte[i])) > 1]
    ci.append({"model": m, "test_auc": roc_auc_score(yte, p),
               "auc_95ci_low": np.percentile(b, 2.5), "auc_95ci_high": np.percentile(b, 97.5)})
pd.DataFrame(ci).round(3).to_csv(os.path.join(RES, "phase2_dependence_test_auc_ci.csv"), index=False)

# stability: 5x5 repeated CV on the TRAINING part with each model's chosen settings (test set stays untouched)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED); rep = []
for m, (best, thr) in fitted.items():
    if m.startswith("Baseline"):
        rep.append({"model": m, "repeated_cv_auc_mean": 0.5, "repeated_cv_auc_sd": 0.0}); continue
    sc = cross_val_score(best, Xtr, ytr, cv=rcv, scoring="roc_auc", n_jobs=-1)
    rep.append({"model": m, "repeated_cv_auc_mean": sc.mean(), "repeated_cv_auc_sd": sc.std()})
pd.DataFrame(rep).round(3).to_csv(os.path.join(RES, "phase2_dependence_repeated_cv.csv"), index=False)

# permutation check on the best-tuned logistic regression: real labels vs shuffled labels
pipe = fitted["Logistic regression"][0]
real = cross_val_score(pipe, Xtr, ytr, cv=rcv, scoring="roc_auc", n_jobs=-1).mean()
shuf = np.random.default_rng(SEED)
perm = [cross_val_score(pipe, Xtr, shuf.permutation(ytr), cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                        scoring="roc_auc", n_jobs=-1).mean() for _ in range(20)]
pd.DataFrame([{"check": "logistic regression", "real_labels_auc": real, "shuffled_mean": np.mean(perm),
               "shuffled_max": np.max(perm)}]).round(3).to_csv(os.path.join(RES, "phase2_dependence_permutation_check.csv"), index=False)

# feature importance for the charts: LR standardised coefficients, tree / forest / boosting importances
imp = {}
for m in ["Logistic regression", "Decision tree", "Random forest", "Gradient boosting"]:
    best = fitted[m][0]
    names = [n.split("__", 1)[1] for n in best.named_steps["pre"].get_feature_names_out()]
    est = best.named_steps["m"]
    vals = est.coef_[0] if m == "Logistic regression" else est.feature_importances_
    s = pd.Series(vals, index=names).sort_values(key=abs, ascending=False)
    s.head(15).round(4).rename("standardised_coefficient" if m == "Logistic regression" else "importance") \
        .to_csv(os.path.join(RES, f"phase2_dependence_top_features_{m.split()[0].lower()}.csv"))
    imp[m] = s

joblib.dump({m: v[0] for m, v in fitted.items()}, os.path.join(MOD, "phase2_dependence_models.joblib"))
print("Done. Results in", RES)
