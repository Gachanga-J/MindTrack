"""
MindTrack - Member A, Phase 2 improvements and sensitivity checks.

Run from the project root:   python src/memberA_phase2_improvements.py [N_REPEATS]
(default 5 repeats of 5-fold CV = 25 folds per model; use 1 for a quick smoke test)

A  Headline: repeated stratified 5-fold CV on ALL 1,220 students, fixed settings (nothing tuned on the folds).
   Metrics suited to an ~18% positive group: ROC AUC, PR AUC (average precision), precision at 70% recall, Brier score.
B  Models added: elastic-net logistic regression, random forest on the top-15 inputs (selection INSIDE each fold),
   soft-voting ensemble, calibrated ensemble, models with a compact phone-dependence score (F3 total, F4 PCA).
C  Beck score itself: ridge / random forest / gradient boosting regression on bdi_total (Spearman, plus AUC of the
   predicted score against the 11+ flag).
D  Sensitivity checks: S1 Beck 17+, S2 item I removed from the total, S3 students under 18 removed.
No bdi_* column is ever an input. Scalers/encoders/selectors are fitted inside each training fold.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor, VotingClassifier)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ONLY_SENS = len(sys.argv) > 2 and sys.argv[2] == "sens"   # rerun only section D
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)

d = pd.read_csv(os.path.join(ROOT, "data", "processed", "model_ready.csv"))
ordinal = ["age", "semester", "ses", "family_expression", "family_activities", "mobile_use_rules",
           "phone_discrimination", "phone_related_arguments", "time_first_cellphone"]
apps = [c for c in d.columns if c.startswith("app_")]
dums = [c for c in d.columns if c.startswith(("department_", "sex_", "marital_status_", "living_arrangement_",
                                                "residence_area_", "subscription_plan_", "network_access_",
                                                "areas_use_", "type_md_"))]
tmd = [f"tmd_q{i}" for i in range(1, 23)]
F1 = ordinal + apps + dums
F2 = F1 + tmd
F3 = F1 + ["tmd_total"]                     # one compact dependence score instead of 22 columns
assert not [c for c in F2 + F3 if c.startswith("bdi_") or "depress" in c or c == "dependence_flag"]
SCALE = ordinal + tmd + ["tmd_total"]


def pre(cols, pca=False):
    sc = [c for c in cols if c in SCALE and not (pca and c in tmd)]
    parts = [("scale", StandardScaler(), sc)]
    if pca:
        parts.append(("pca", Pipeline([("s", StandardScaler()), ("p", PCA(3, random_state=SEED))]), tmd))
    return ColumnTransformer(parts, remainder="passthrough")


def P(cols, model, pca=False):
    return Pipeline([("pre", pre(cols, pca)), ("m", model)])


LR = lambda: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
RF = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=1)
GB = lambda: GradientBoostingClassifier(max_depth=2, learning_rate=0.05, n_estimators=100, subsample=0.8, random_state=SEED)
EN = lambda: LogisticRegression(C=0.1, l1_ratio=0.5, solver="saga", max_iter=5000, class_weight="balanced")


def classifiers(cols, full=True):
    m = {"Baseline (most common)": P(cols, DummyClassifier(strategy="most_frequent")),
         "Logistic regression": P(cols, LR()), "Random forest": P(cols, RF()), "Gradient boosting": P(cols, GB())}
    if not full:
        return m
    m["Decision tree"] = P(cols, DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=SEED))
    m["Elastic-net logistic"] = P(cols, EN())
    sel = SelectFromModel(LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=5000, class_weight="balanced"),
                          max_features=15, threshold=-np.inf)
    m["Random forest, top-15 inputs"] = Pipeline([("pre", pre(cols)), ("sel", sel), ("m", RF())])
    ens = lambda: VotingClassifier([("lr", P(cols, LR())), ("rf", P(cols, RF())), ("gb", P(cols, GB()))], voting="soft")
    m["Ensemble (LR+RF+GB)"] = ens()
    m["Ensemble, calibrated"] = CalibratedClassifierCV(ens(), method="sigmoid", cv=3)
    return m


def prec_at_recall(y, p, target=0.70):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr[rc >= target]
    return float(ok.max()) if len(ok) else 0.0


def cv_eval(models, X, y, label, extra):
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED)
    folds = list(rcv.split(X, y)); out = []
    for name, est in models.items():
        a, ap, pr70, br = [], [], [], []
        for tr, te in folds:
            if name.startswith("Baseline"):
                a.append(0.5); ap.append(y[te].mean()); pr70.append(y[te].mean()); br.append(np.nan); continue
            from sklearn.base import clone
            e = clone(est).fit(X.iloc[tr], y[tr]); p = e.predict_proba(X.iloc[te])[:, 1]
            a.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p))
            pr70.append(prec_at_recall(y[te], p)); br.append(brier_score_loss(y[te], p))
        r = {**extra, "model": name, "auc_mean": np.mean(a), "auc_sd": np.std(a), "pr_auc_mean": np.mean(ap),
             "precision_at_70pct_recall": np.mean(pr70), "brier": np.nanmean(br) if not name.startswith("Baseline") else np.nan,
             "prevalence": y.mean(), "n_flagged": int(y.sum()), "n": len(y)}
        out.append(r); print(f"{label:34s} {name:30s} AUC {r['auc_mean']:.3f}±{r['auc_sd']:.3f}  PR-AUC {r['pr_auc_mean']:.3f}  P@R70 {r['precision_at_70pct_recall']:.3f}", flush=True)
    return out


y_main = d["depressed_flag"].values
if not ONLY_SENS:
    # ------------------------------------------------------------------ A + B headline
    rows = []
    for sname, cols in {"F1 phone & background": F1, "F2 + 22 dependence questions": F2}.items():
        rows += cv_eval(classifiers(cols), d[cols], y_main, f"HEADLINE {sname[:2]}", {"feature_set": sname})
    rows += cv_eval({k: v for k, v in classifiers(F3).items() if k in ("Logistic regression", "Random forest", "Elastic-net logistic")},
                    d[F3], y_main, "HEADLINE F3", {"feature_set": "F3 + dependence total (1 column)"})
    F4 = F2
    rows += cv_eval({"Logistic regression": P(F4, LR(), pca=True), "Random forest": P(F4, RF(), pca=True)},
                    d[F4], y_main, "HEADLINE F4", {"feature_set": "F4 + 3 PCA components of the 22 questions"})
    head = pd.DataFrame(rows); head.round(3).to_csv(os.path.join(RES, "phase2b_headline_repeated_cv.csv"), index=False)

    # ------------------------------------------------------------------ C regression on the Beck score
    reg_rows = []
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=SEED)
    score = d["bdi_total"].values.astype(float)
    from sklearn.base import clone
    for sname, cols in {"F1 phone & background": F1, "F2 + 22 dependence questions": F2}.items():
        regs = {"Ridge": P(cols, Ridge(alpha=300)),
                "Random forest regressor": P(cols, RandomForestRegressor(n_estimators=200, min_samples_leaf=15, random_state=SEED, n_jobs=1)),
                "Gradient boosting regressor": P(cols, GradientBoostingRegressor(max_depth=2, learning_rate=0.05, n_estimators=100, subsample=0.8, random_state=SEED))}
        folds = list(rcv.split(d[cols], y_main))
        for name, est in regs.items():
            sp, au, ap = [], [], []
            for tr, te in folds:
                e = clone(est).fit(d[cols].iloc[tr], np.log1p(score[tr])); p = e.predict(d[cols].iloc[te])
                sp.append(spearmanr(p, score[te]).correlation); au.append(roc_auc_score(y_main[te], p)); ap.append(average_precision_score(y_main[te], p))
            reg_rows.append({"feature_set": sname, "model": name, "spearman_mean": np.mean(sp), "spearman_sd": np.std(sp),
                             "auc_for_flag_11plus": np.mean(au), "auc_sd": np.std(au), "pr_auc_for_flag": np.mean(ap)})
            print(f"REGRESSION {sname[:2]} {name:28s} Spearman {np.mean(sp):.3f}  AUC(flag) {np.mean(au):.3f}", flush=True)
    pd.DataFrame(reg_rows).round(3).to_csv(os.path.join(RES, "phase2b_beck_score_regression.csv"), index=False)


# ------------------------------------------------------------------ D sensitivity checks
checks = {"MAIN: Beck 11+ (all 1,220)": (d, d["depressed_flag"].values),
          "S1: Beck 17+": (d, d["depressed_flag_ge17"].values),
          "S2: item I removed from total": (d, d["depressed_flag_excl_item_i"].values),
          "S3: under-18s removed": (d[d.age >= 18].reset_index(drop=True), d.loc[d.age >= 18, "depressed_flag"].values)}
s_rows = []
for cname, (dd, yy) in checks.items():
    for sname, cols in {"F1": F1, "F2": F2}.items():
        s_rows += cv_eval(classifiers(cols, full=False), dd[cols], yy, f"{cname[:3]} {sname}", {"check": cname, "feature_set": sname})
sens = pd.DataFrame(s_rows); sens.round(3).to_csv(os.path.join(RES, "phase2b_sensitivity_checks.csv"), index=False)
base = sens[sens.check.str.startswith("MAIN")].set_index(["feature_set", "model"])["auc_mean"]
sens["auc_change_vs_main"] = [r.auc_mean - base[(r.feature_set, r.model)] for r in sens.itertuples()]
sens[~sens.model.str.startswith("Baseline")].groupby("check")["auc_change_vs_main"].agg(["mean", "min", "max"]).round(3)\
    .to_csv(os.path.join(RES, "phase2b_sensitivity_summary.csv"))

facts = pd.DataFrame([{"fact": "flagged, Beck 11+", "value": int(d.depressed_flag.sum())},
                      {"fact": "flagged, Beck 17+", "value": int(d.depressed_flag_ge17.sum())},
                      {"fact": "flagged, item I removed", "value": int(d.depressed_flag_excl_item_i.sum())},
                      {"fact": "students changing group without item I", "value": int((d.depressed_flag != d.depressed_flag_excl_item_i).sum())},
                      {"fact": "students under 18", "value": int((d.age < 18).sum())},
                      {"fact": "under-18s flagged (11+)", "value": int(d.loc[d.age < 18, "depressed_flag"].sum())},
                      {"fact": "age range of under-18s", "value": f"{d.age[d.age < 18].min()}-{d.age[d.age < 18].max()}"}])
facts.to_csv(os.path.join(RES, "phase2b_sensitivity_facts.csv"), index=False)
print(facts.to_string(index=False)); print("Done.")
