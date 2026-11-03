"""
MindTrack - Member C, Phase 2: feature-selection step used by the app-ready models.

It is a scikit-learn "transformer", which means it can sit INSIDE a Pipeline. That matters:
when cross-validation holds out a fold, the selection is re-done using ONLY the training part of
that fold, so the held-out students never influence which inputs are kept (no leakage).

Two stages (either can be switched off):
  1. CORRELATION filter (corr_cut)  - look for pairs of inputs that say almost the same thing
     (|correlation| >= corr_cut). From each such pair keep the one that is more related to the target.
  2. IMPORTANCE filter (top_k)      - fit a small random forest on what is left and keep the top_k
     inputs with the highest importance.

With corr_cut=None and top_k=None it keeps everything (used for the "all app inputs" variant).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class CorrImportanceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, corr_cut=None, top_k=None, seed=42):
        self.corr_cut = corr_cut
        self.top_k = top_k
        self.seed = seed

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        cols = list(X.columns)

        # Stage 1: correlation filter (greedy; strongest link to the target is considered first)
        if self.corr_cut is not None:
            to_target = X.corrwith(y).abs().fillna(0).sort_values(ascending=False)
            between = X.corr().abs().fillna(0)
            kept = []
            for c in to_target.index:
                if all(between.loc[c, k] < self.corr_cut for k in kept):
                    kept.append(c)
            cols = [c for c in cols if c in set(kept)]
        self.kept_after_corr_ = list(cols)

        # Stage 2: importance filter (small random forest on the survivors)
        self.importances_ = None
        if self.corr_cut is not None or self.top_k is not None:
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                        class_weight="balanced", random_state=self.seed, n_jobs=-1)
            rf.fit(X[cols], y)
            self.importances_ = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
        if self.top_k is not None and len(cols) > self.top_k:
            top = set(self.importances_.index[: self.top_k])
            cols = [c for c in cols if c in top]
        self.cols_ = list(cols)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.cols_]
