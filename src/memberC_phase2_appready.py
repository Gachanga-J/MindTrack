"""
MindTrack - Member C, Phase 2: app-ready models and feature selection (script version).

This file is generated from notebooks/memberC_phase2_appready.ipynb (same code, no explanations).
Use the notebook to learn and present; use this script to re-run everything in one go:

    python src/memberC_phase2_appready.py        # from the project root

It reads data/processed/train.csv and test.csv (Member C's shared split) and the repeated-CV files of
Members A and B, and writes results/phase2_appready_*.csv, docs/figures/*.png, docs/app_inputs_memberC.csv
and models/phase2_appready_final_models.joblib.
Progress is checkpointed in .cache_appready/ (delete that folder to force a completely fresh run).
"""
import matplotlib
matplotlib.use("Agg")          # draw charts to files without opening windows


# ===== MindTrack | Phase 2 (Member C): "App-ready" models and feature selection =====


# ===== Step 0: Setup and the shared split =====

# ------------------------------ cell 2 ------------------------------
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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

# Works whether you run from the project root or from the notebooks/ folder
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from memberC_selector import CorrImportanceSelector     # our feature-selection step (src/memberC_selector.py)

PROC, RES, MOD, DOCS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models", ROOT / "docs"
FIG = DOCS / "figures"
CACHE = ROOT / ".cache_appready"          # speed-up + checkpoints; safe to delete, ignored by git (see .gitignore note)
for p in (RES, MOD, FIG, CACHE):
    p.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(PROC / "train.csv")      # NOT the scaled copies: scaling is done inside each CV fold instead
test = pd.read_csv(PROC / "test.csv")
assert not set(train["code"]) & set(test["code"]), "train and test students overlap!"

TARGETS = {"Depression flag": "depressed_flag", "Dependence flag": "dependence_flag"}
for name, t in TARGETS.items():
    print(f"{name:16s} train {len(train)} ({train[t].sum()} flagged, {train[t].mean():.1%}) | "
          f"test {len(test)} ({test[t].sum()} flagged, {test[t].mean():.1%})")


# ===== Step 1: Define the inputs the app can collect =====

# ------------------------------ cell 4 ------------------------------
ORDERED = ["age", "semester", "ses", "time_first_cellphone"]      # numbers / ordered levels (these get scaled)
APPS = [c for c in train.columns if c.startswith("app_")]         # 8 yes/no flags
DUMMY_PREFIXES = ("department_", "sex_", "living_arrangement_", "residence_area_",
                  "subscription_plan_", "network_access_", "areas_use_", "type_md_")
DUMMIES = [c for c in train.columns if c.startswith(DUMMY_PREFIXES)]   # yes/no columns made in Phase 1 (one-hot)
APP_INPUTS = ORDERED + APPS + DUMMIES

# ---- the security guard -------------------------------------------------------------------------------------
BANNED_PREFIXES = ("tmd_", "bdi_", "marital_status_", "pos_", "time_stand", "time_walk", "time_sit", "time_lying",
                   "family_", "mobile_use_rules", "phone_discrimination", "phone_related_arguments",
                   "avg_daily_use", "usual_phone_location", "primary_input", "frequent_touch", "hand_dominance",
                   "program", "age_group", "code")
banned = [c for c in APP_INPUTS if c.startswith(BANNED_PREFIXES) or "depress" in c or "dependence" in c]
assert not banned, f"Banned columns on the input list: {banned}"
assert len(APP_INPUTS) == len(set(APP_INPUTS)) == 31
print(len(APP_INPUTS), "app inputs:", len(ORDERED), "ordered/numeric,", len(APPS), "app flags,", len(DUMMIES), "yes/no background columns")

# A small, readable table of the inputs (also saved for the report and for Phase 3)
group_of = lambda c: ("Apps" if c.startswith("app_") else
                      "Phone" if c.startswith(("subscription_plan_", "network_access_", "areas_use_", "type_md_")) or c == "time_first_cellphone"
                      else "About the student")
inputs_table = pd.DataFrame({"input": APP_INPUTS, "group": [group_of(c) for c in APP_INPUTS],
                             "kind": ["numeric / ordered (scaled)" if c in ORDERED else "yes/no (0 or 1)" for c in APP_INPUTS]})
inputs_table.to_csv(DOCS / "app_inputs_memberC.csv", index=False)
display(inputs_table.groupby(["group", "kind"]).size().rename("how many").reset_index())


# ===== Step 2: Feature selection, part 1 (correlation) =====

# ------------------------------ cell 6 ------------------------------
# (a) correlation of every app input with each target (sign shows direction, size shows strength)
corr_target = pd.DataFrame({n: train[APP_INPUTS].corrwith(train[t]) for n, t in TARGETS.items()})
corr_target["strongest_abs"] = corr_target.abs().max(axis=1)
corr_target = corr_target.sort_values("strongest_abs", ascending=False)
corr_target.round(3).to_csv(RES / "phase2_appready_feature_correlation_with_target.csv", index_label="input")
print("Strongest correlations with the targets (note how small they are: nothing above ~0.13)")
display(corr_target.head(8).round(3))

# (b) pairs of inputs that repeat each other
CORR_CUT = 0.8
cm = train[APP_INPUTS].corr()
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(APP_INPUTS) for b in APP_INPUTS[i + 1:] if abs(cm.loc[a, b]) >= CORR_CUT]
redundant = pd.DataFrame(pairs, columns=["input_1", "input_2", "correlation"]).sort_values("correlation", key=abs, ascending=False)
redundant.round(3).to_csv(RES / "phase2_appready_redundant_pairs.csv", index=False)
print(f"\nPairs of inputs with |correlation| >= {CORR_CUT} (near-duplicates):")
display(redundant.round(3))

# chart: top 10 inputs by correlation with each target
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for ax, (n, t) in zip(axes, TARGETS.items()):
    s = corr_target[n].reindex(corr_target[n].abs().sort_values(ascending=False).index).head(10)[::-1]
    ax.barh(s.index, s.values, color=["tab:red" if v > 0 else "tab:blue" for v in s.values])
    ax.axvline(0, color="black", lw=0.8); ax.set_title(f"{n}: top 10 correlations (train)")
    ax.set_xlabel("correlation (red = higher value goes with flagged)")
plt.tight_layout(); plt.savefig(FIG / "appready_correlation_with_targets.png", dpi=130); plt.close('all')


# ===== **What this tells us** =====


# ===== Step 3: Build the app-ready models =====

# ------------------------------ cell 9 ------------------------------
SCALE_PATTERN = "^(" + "|".join(ORDERED) + ")$"      # only the 4 numeric columns are scaled; yes/no columns are left alone


def make_pipe(model, selector):
    pre = ColumnTransformer([("scale", StandardScaler(), make_column_selector(pattern=SCALE_PATTERN))],
                            remainder="passthrough")
    return Pipeline([("sel", selector), ("pre", pre), ("m", model)], memory=str(CACHE / "pipe"))


def zoo(variant):
    """The models and the settings we try. 'selected' also tries k = 8, 12 or 18 inputs."""
    selector = CorrImportanceSelector() if variant == "All app inputs" else CorrImportanceSelector(corr_cut=CORR_CUT, seed=SEED)
    extra = {} if variant == "All app inputs" else {"sel__top_k": [8, 12, 18]}
    mk = lambda m: make_pipe(m, selector)
    return {
        "Baseline (most common)": (mk(DummyClassifier(strategy="most_frequent")), {}),
        "Logistic regression": (mk(LogisticRegression(max_iter=3000, class_weight="balanced")),
                                {"m__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1], **extra}),
        "Decision tree": (mk(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
                          {"m__max_depth": [2, 3, 4, 5], "m__min_samples_leaf": [10, 20, 40], **extra}),
        "Random forest": (mk(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
                          {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [5, 10, 20], **extra}),
        "Gradient boosting": (mk(GradientBoostingClassifier(random_state=SEED)),
                              {"m__max_depth": [1, 2, 3], "m__learning_rate": [0.03, 0.1], "m__n_estimators": [50, 150], **extra}),
    }


def scores(y, pred, prob):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob)}


def pick_threshold(y, prob):
    """Same rule for every model (as Members A and B): the cut-off that maximises balanced accuracy on out-of-fold TRAIN predictions."""
    fpr, tpr, thr = roc_curve(y, prob)
    return float(thr[np.argmax(tpr - fpr)])

# ------------------------------ cell 10 ------------------------------
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
VARIANTS = ["All app inputs", "Selected (correlation + importance)"]
rows, cms, fitted, test_pred = [], [], {}, {}
Xtr, Xte = train[APP_INPUTS], test[APP_INPUTS]

for tname, tcol in TARGETS.items():
    ytr, yte = train[tcol].values, test[tcol].values
    for variant in VARIANTS:
        vkey = "All app inputs" if variant.startswith("All") else "Selected"
        ckpt = CACHE / f"{tcol}__{vkey.replace(' ', '_')}.joblib"
        if ckpt.exists():                                  # already done in an earlier run: reload instead of recomputing
            b_rows, b_cms, b_fit, b_pred = joblib.load(ckpt)
            rows += b_rows; cms += b_cms; fitted.update(b_fit); test_pred.update(b_pred)
            print(f"=== {tname} | {variant}: loaded from checkpoint ===", flush=True); continue
        n0 = len(rows); keys0 = set(fitted)
        print(f"\n=== {tname} | {variant} ===", flush=True)
        for mname, (pipe, grid) in zoo("All app inputs" if variant.startswith("All") else "Selected").items():
            if grid:
                gs = GridSearchCV(pipe, grid, cv=cv5, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
                best, params, cv_auc = gs.best_estimator_, gs.best_params_, gs.best_score_
                cv_sd = gs.cv_results_["std_test_score"][gs.best_index_]
                oof = cross_val_predict(best, Xtr, ytr, cv=cv5, method="predict_proba")[:, 1]
                thr = pick_threshold(ytr, oof)
            else:
                best = pipe.fit(Xtr, ytr); params, cv_auc, cv_sd, thr = {}, 0.5, 0.0, 0.5
            ptr, pte = best.predict_proba(Xtr)[:, 1], best.predict_proba(Xte)[:, 1]
            if not grid:                                     # baseline: says "not flagged" for everyone
                ctr, cte = np.zeros_like(ytr), np.zeros_like(yte)
            else:
                ctr, cte = (ptr >= thr).astype(int), (pte >= thr).astype(int)
            mtr, mte = scores(ytr, ctr, ptr), scores(yte, cte, pte)
            n_in = len(best.named_steps["sel"].cols_)
            r = {"target": tname, "inputs": variant, "model": mname, "n_inputs_used": n_in,
                 "cv5_auc_mean(train)": cv_auc, "cv5_auc_sd": cv_sd, "threshold": thr,
                 "train_auc": mtr["auc"], "train_balanced_acc": mtr["balanced_accuracy"]}
            r.update({f"test_{k}": v for k, v in mte.items()})
            r["auc_gap_train_minus_test"] = mtr["auc"] - mte["auc"]
            r["best_params"] = json.dumps({k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()})
            rows.append(r); fitted[(tname, variant, mname)] = (best, thr); test_pred[(tname, variant, mname)] = (pte, cte)
            tn, fp, fn, tp = confusion_matrix(yte, cte, labels=[0, 1]).ravel()
            cms.append({"target": tname, "inputs": variant, "model": mname, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
            print(f"{mname:24s} inputs {n_in:2d} | cv5 AUC {cv_auc:.3f} | test AUC {mte['auc']:.3f} | "
                  f"recall {mte['recall']:.2f} | prec {mte['precision']:.2f} | gap {r['auc_gap_train_minus_test']:.2f}", flush=True)
        new = [k for k in fitted if k not in keys0]
        joblib.dump((rows[n0:], cms[-len(new):], {k: fitted[k] for k in new}, {k: test_pred[k] for k in new}), ckpt)

res = pd.DataFrame(rows)
res.round(3).to_csv(RES / "phase2_appready_results.csv", index=False)
pd.DataFrame(cms).to_csv(RES / "phase2_appready_confusion_matrices.csv", index=False)


# ===== **Reading the results table.** `cv5_auc_mean(train)` is the practice-quiz score, `test_*` is the exam, and `auc_gap_train_minus_test` shows memorising: a big gap means the model learned the textbook by heart (over-fitting). `n_inputs_used` shows how many inputs the model kept. =====

# ------------------------------ cell 12 ------------------------------
cols = ["target", "inputs", "model", "n_inputs_used", "test_accuracy", "test_precision", "test_recall", "test_f1",
        "test_balanced_accuracy", "test_auc", "train_auc", "auc_gap_train_minus_test"]
for tname in TARGETS:
    print(f"\n{tname} - test set (244 students)")
    display(res.loc[res.target == tname, cols[1:]].round(2).reset_index(drop=True))


# ===== Step 4: Honesty checks =====

# ------------------------------ cell 14 ------------------------------
rng = np.random.default_rng(SEED)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
ci_rows, rep_rows, perm_rows = [], [], []

for (tname, variant, mname), (best, thr) in fitted.items():
    if mname.startswith("Baseline"):
        rep_rows.append({"target": tname, "inputs": variant, "model": mname, "repeated_cv_auc_mean": 0.5, "repeated_cv_auc_sd": 0.0})
        continue
    tcol = TARGETS[tname]; ytr, yte = train[tcol].values, test[tcol].values
    pte = test_pred[(tname, variant, mname)][0]
    boots = [roc_auc_score(yte[i], pte[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(1000)) if len(set(yte[i])) > 1]
    ci_rows.append({"target": tname, "inputs": variant, "model": mname, "test_auc": roc_auc_score(yte, pte),
                    "auc_95ci_low": np.percentile(boots, 2.5), "auc_95ci_high": np.percentile(boots, 97.5)})
    sc = cross_val_score(best, Xtr, ytr, cv=rcv, scoring="roc_auc", n_jobs=-1)
    rep_rows.append({"target": tname, "inputs": variant, "model": mname,
                     "repeated_cv_auc_mean": sc.mean(), "repeated_cv_auc_sd": sc.std()})

# shuffled-label check on the (tuned) logistic regression of each target and input version
for tname, tcol in TARGETS.items():
    ytr = train[tcol].values
    for variant in VARIANTS:
        pipe = fitted[(tname, variant, "Logistic regression")][0]
        real = cross_val_score(pipe, Xtr, ytr, cv=rcv, scoring="roc_auc", n_jobs=-1).mean()
        shuf = np.random.default_rng(SEED)
        perm = [cross_val_score(pipe, Xtr, shuf.permutation(ytr),
                                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED),
                                scoring="roc_auc", n_jobs=-1).mean() for _ in range(10)]
        perm_rows.append({"target": tname, "inputs": variant, "model": "Logistic regression", "real_labels_auc": real,
                          "shuffled_mean": np.mean(perm), "shuffled_max": np.max(perm)})

ci, rep, perm = pd.DataFrame(ci_rows), pd.DataFrame(rep_rows), pd.DataFrame(perm_rows)
ci.round(3).to_csv(RES / "phase2_appready_test_auc_ci.csv", index=False)
rep.round(3).to_csv(RES / "phase2_appready_repeated_cv.csv", index=False)
perm.round(3).to_csv(RES / "phase2_appready_permutation_check.csv", index=False)

print("Repeated 5x5 cross-validation AUC (mean +/- sd) - the main result")
display(rep.pivot_table(index=["target", "model"], columns="inputs", values="repeated_cv_auc_mean").round(3))
print("Test AUC with 95% bootstrap range")
display(ci.round(2))
print("Shuffled-label check (real labels should beat shuffled ones clearly)")
display(perm.round(3))


# ===== Step 5: Feature selection, part 2 (importance) =====

# ------------------------------ cell 16 ------------------------------
SIMPLICITY = {"Logistic regression": 0, "Decision tree": 1, "Random forest": 2, "Gradient boosting": 3}
cand = rep[~rep.model.str.startswith("Baseline")].merge(res[["target", "inputs", "model", "n_inputs_used"]], on=["target", "inputs", "model"])
final_rows, FINAL = [], {}
for tname in TARGETS:
    c = cand[cand.target == tname].copy()
    top = c.repeated_cv_auc_mean.max()
    tied = c[c.repeated_cv_auc_mean >= top - 0.01].copy()
    tied["simple"] = tied.model.map(SIMPLICITY)
    pick = tied.sort_values(["n_inputs_used", "simple", "repeated_cv_auc_mean"], ascending=[True, True, False]).iloc[0]
    FINAL[tname] = (pick.inputs, pick.model)
    final_rows.append({"target": tname, "inputs": pick.inputs, "model": pick.model, "n_inputs_used": pick.n_inputs_used,
                       "repeated_cv_auc_mean": pick.repeated_cv_auc_mean, "repeated_cv_auc_sd": pick.repeated_cv_auc_sd,
                       "best_repeated_cv_auc_overall": top, "models_tied_with_best": len(tied)})
final = pd.DataFrame(final_rows)
final.round(3).to_csv(RES / "phase2_appready_final_choice.csv", index=False)
display(final.round(3))

# ------------------------------ cell 17 ------------------------------
imp_rows, sel_rows = [], []
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (tname, tcol) in zip(axes, TARGETS.items()):
    ytr, yte = train[tcol].values, test[tcol].values
    variant, mname = FINAL[tname]
    best, thr = fitted[(tname, variant, mname)]
    sel = best.named_steps["sel"]

    # selector's view (only meaningful when the selector was active)
    if sel.importances_ is not None:
        for rank, (feat, v) in enumerate(sel.importances_.items(), 1):
            sel_rows.append({"target": tname, "model": mname, "rank": rank, "input": feat,
                             "forest_importance": v, "kept": feat in sel.cols_})
    # permutation importance on the test set (drop in AUC when one input is shuffled)
    pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED, n_jobs=1)
    s = pd.DataFrame({"input": APP_INPUTS, "auc_drop_mean": pi.importances_mean, "auc_drop_sd": pi.importances_std,
                      "used_by_model": [c in sel.cols_ for c in APP_INPUTS]}).sort_values("auc_drop_mean", ascending=False)
    s.insert(0, "target", tname); s.insert(1, "model", f"{mname} ({variant})")
    imp_rows.append(s)
    top = s.head(10)[::-1]
    ax.barh(top.input, top.auc_drop_mean, xerr=top.auc_drop_sd, color="tab:green")
    ax.set_title(f"{tname}: {mname}\n({sel.cols_.__len__()} inputs used)"); ax.set_xlabel("drop in test AUC when the input is shuffled")
plt.tight_layout(); plt.savefig(FIG / "appready_permutation_importance.png", dpi=130); plt.close('all')

imp_all = pd.concat(imp_rows); imp_all.round(4).to_csv(RES / "phase2_appready_permutation_importance.csv", index=False)
if sel_rows:
    pd.DataFrame(sel_rows).round(4).to_csv(RES / "phase2_appready_selector_importance.csv", index=False)

# which inputs does each "Selected" model keep? (fitted on the whole training set)
keep_rows = []
for (tname, variant, mname), (best, thr) in fitted.items():
    if variant.startswith("Selected") and not mname.startswith("Baseline"):
        sel = best.named_steps["sel"]
        keep_rows.append({"target": tname, "model": mname, "n_kept": len(sel.cols_),
                          "dropped_as_duplicates": ", ".join(sorted(set(APP_INPUTS) - set(sel.kept_after_corr_))),
                          "kept_inputs": ", ".join(sel.cols_)})
keep = pd.DataFrame(keep_rows); keep.to_csv(RES / "phase2_appready_selected_inputs.csv", index=False)
print("Near-duplicates dropped by the correlation filter (fitted on the full training set; the filter keeps the member of a pair that is more related to the target, so the list can differ by target):")
for t_, g in keep.groupby("target"):
    print(f"  {t_}: {g['dropped_as_duplicates'].iloc[0]}")
for tname in TARGETS:
    v, m = FINAL[tname]
    print(f"\nTop 8 inputs for the final {tname} model ({m}, {v}):")
    display(imp_all[imp_all.target == tname].head(8)[["input", "auc_drop_mean", "auc_drop_sd", "used_by_model"]].round(4))


# ===== Step 6: What did "app-ready" cost us? =====

# ------------------------------ cell 19 ------------------------------
a = pd.read_csv(RES / "phase2_depression_repeated_cv.csv")
a = a[a.feature_set.str.startswith("F1")][["model", "repeated_cv_auc_mean"]].assign(target="Depression flag")
b = pd.read_csv(RES / "phase2_dependence_repeated_cv.csv")[["model", "repeated_cv_auc_mean"]].assign(target="Dependence flag")
f1 = pd.concat([a, b]).rename(columns={"repeated_cv_auc_mean": "A/B larger input list (F1)"})
mine = rep.pivot_table(index=["target", "model"], columns="inputs", values="repeated_cv_auc_mean").reset_index()
cmp_ = mine.merge(f1, on=["target", "model"], how="left")
cmp_["cost_of_app_only_inputs"] = cmp_["All app inputs"] - cmp_["A/B larger input list (F1)"]
cmp_ = cmp_[~cmp_.model.str.startswith("Baseline")]
cmp_.round(3).to_csv(RES / "phase2_appready_vs_F1.csv", index=False)
display(cmp_.round(3))

fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27
lab = [f"{t.split()[0]}\n{m}" for t, m in zip(cmp_.target, cmp_.model)]; x = np.arange(len(cmp_))
ax.bar(x - w, cmp_["A/B larger input list (F1)"], w, label="A/B larger list (F1)")
ax.bar(x, cmp_["All app inputs"], w, label="App-ready: all 31")
ax.bar(x + w, cmp_["Selected (correlation + importance)"], w, label="App-ready: selected")
ax.axhline(0.5, color="black", ls="--", lw=0.8, label="baseline / coin flip")
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0.45, 0.72); ax.set_ylabel("repeated-CV AUC (train)")
ax.set_title("What does using only app-collectable inputs cost?"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "appready_vs_F1.png", dpi=130); plt.close('all')


# ===== Step 7: Save everything for the team and for Phase 3 =====

# ------------------------------ cell 21 ------------------------------
pr = test[["code"]].copy(); pr["depressed_flag"] = test["depressed_flag"].values; pr["dependence_flag"] = test["dependence_flag"].values
for (tname, variant, mname), (pte, cte) in test_pred.items():
    key = f"{tname.split()[0]}|{'all' if variant.startswith('All') else 'selected'}|{mname}"
    pr[f"{key}|prob"] = pte; pr[f"{key}|pred"] = cte
pr.round(4).to_csv(RES / "phase2_appready_test_predictions.csv", index=False)

joblib.dump({t: {"inputs": v, "model_name": m, "pipeline": fitted[(t, v, m)][0], "threshold": fitted[(t, v, m)][1],
                 "input_columns": APP_INPUTS}
             for t, (v, m) in FINAL.items()}, MOD / "phase2_appready_final_models.joblib")
print("Saved. Files now in results/ for this section:")
print(sorted(p.name for p in RES.glob("phase2_appready_*")))


# ===== Step 8: Findings in plain language (for the report and the presentation) =====
