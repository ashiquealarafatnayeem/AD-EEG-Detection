import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent))
from eegio import PROCESSED

TASKS = {
    "AD_vs_CN":  ["CN", "AD"],
    "FTD_vs_CN": ["CN", "FTD"],
    "AD_vs_FTD": ["FTD", "AD"],
    "3class":    ["CN", "FTD", "AD"],
}
FEATURE_SETS = ["region", "channel"]
MODELS = ["dummy", "logreg", "linsvm", "rbfsvm", "rf", "knn"]
MODEL_NAMES = {"dummy": "Chance", "logreg": "Logistic regression",
               "linsvm": "Linear SVM", "rbfsvm": "RBF SVM", "rf": "Random forest",
               "knn": "k-nearest neighbours"}
PRIMARY = ("AD_vs_CN", "region", "logreg")
META = ["participant_id", "group", "age", "sex", "n_epochs"]
N_FOLDS, N_REPEATS, INNER_FOLDS = 5, 10, 5
N_BOOT = 2000
SEED = 2026

def load_task(task, feature_set):
    classes = TASKS[task]
    source = "region" if feature_set == "age_sex" else feature_set
    df = pd.read_csv(PROCESSED / f"features_{source}.csv")
    df = df[df.group.isin(classes)].reset_index(drop=True)
    if feature_set == "age_sex":
        X = pd.DataFrame({"age": df.age.astype(float),
                          "sex_male": (df.sex.astype(str).str.upper() == "M").astype(float)})
    else:
        X = df.drop(columns=[c for c in META if c in df.columns]).astype(float)
    y = df.group.map({g: i for i, g in enumerate(classes)}).to_numpy()
    return X.to_numpy(), y, df.participant_id.to_numpy(), list(X.columns), classes

class ConfoundRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, n_conf=2):
        self.n_conf = n_conf

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        C = np.column_stack([np.ones(len(X)), X[:, -self.n_conf:]])
        return X[:, :-self.n_conf], C

    def fit(self, X, y=None):
        F, C = self._split(X)
        self.beta_ = np.linalg.lstsq(C, F, rcond=None)[0]
        return self

    def transform(self, X):
        F, C = self._split(X)
        return F - C @ self.beta_

def model_spec(name, n_features, n_confounds=0):
    if name == "dummy":
        return Pipeline([("clf", DummyClassifier(strategy="constant", constant=0))]), None
    k_grid = [k for k in (10, 25, 50, 100) if k < n_features] + ["all"]
    clf, grid = {
        "logreg": (LogisticRegression(class_weight="balanced", max_iter=5000),
                   {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}),
        "linsvm": (SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0),
                   {"clf__C": [0.001, 0.01, 0.1, 1, 10]}),
        "rbfsvm": (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0),
                   {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale"]}),
        "rf": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
               {"clf__max_features": ["sqrt", 0.3], "clf__min_samples_leaf": [1, 3]}),
        "knn": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7, 9, 15]}),
    }[name]
    if name == "rf":
        k_grid = [k for k in k_grid if k == "all" or k == 25]
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if n_confounds:
        steps.append(("deconfound", ConfoundRegressor(n_conf=n_confounds)))
    steps += [("scale", StandardScaler()), ("select", SelectKBest(f_classif)), ("clf", clf)]
    pipe = Pipeline(steps)
    return pipe, {**grid, "select__k": k_grid}

def run_cv(X, y, name, n_repeats=N_REPEATS, seed=SEED, n_jobs=-1, inner_folds=INNER_FOLDS,
           n_confounds=0):
    warnings.filterwarnings("ignore")
    n_classes = len(np.unique(y))
    outer = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=seed)
    proba = np.full((n_repeats, len(y), n_classes), np.nan)
    pred = np.full((n_repeats, len(y)), -1)
    n_feat = X.shape[1] - n_confounds
    sel = np.zeros(n_feat)
    coef = np.zeros(n_feat)
    params = []
    scoring = "roc_auc" if n_classes == 2 else "balanced_accuracy"
    for f, (tr, te) in enumerate(outer.split(X, y)):
        pipe, grid = model_spec(name, n_feat, n_confounds)
        if grid is None:
            est = pipe.fit(X[tr], y[tr])
        else:
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + f)
            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=n_jobs)
            est = gs.fit(X[tr], y[tr]).best_estimator_
            params.append(gs.best_params_)
            mask = est.named_steps["select"].get_support()
            sel += mask
            c = getattr(est.named_steps["clf"], "coef_", None)
            if c is not None and n_classes == 2 and not hasattr(c, "toarray"):
                coef[mask] += np.ravel(c)
        proba[f // N_FOLDS, te] = est.predict_proba(X[te])
        pred[f // N_FOLDS, te] = est.predict(X[te])
    n_outer = N_FOLDS * n_repeats
    return {"proba": proba, "pred": pred, "params": params,
            "sel_freq": sel / n_outer, "coef": coef / n_outer}

def _auc_binary(y, s):
    r = rankdata(s, axis=1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)

def _scores(y, proba, pred):
    k = proba.shape[2]
    recall = np.stack([(pred[:, y == c] == c).mean(axis=1) for c in range(k)], axis=1)
    out = {"bacc": recall.mean(axis=1)}
    if k == 2:
        out["auc"] = _auc_binary(y, proba[:, :, 1])
        out["sens"], out["spec"] = recall[:, 1], recall[:, 0]
    else:
        out["auc"] = np.mean([_auc_binary((y == c).astype(int), proba[:, :, c])
                              for c in range(k)], axis=0)
    return out

def summarise(y, proba, pred, n_boot=N_BOOT, seed=0):
    s = _scores(y, proba, pred)
    point = {k: float(v.mean()) for k, v in s.items()}
    point["auc_sd_repeats"] = float(s["auc"].std())
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    boot = {"auc": np.empty(n_boot), "bacc": np.empty(n_boot)}
    for i in range(n_boot):
        b = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_class])
        sb = _scores(y[b], proba[:, b], pred[:, b])
        for k in boot:
            boot[k][i] = sb[k].mean()
    for k in boot:
        point[f"{k}_lo"], point[f"{k}_hi"] = (float(v) for v in np.percentile(boot[k], [2.5, 97.5]))
    return point

def mean_confusion(y, pred, n_classes):
    cms = [confusion_matrix(y, p, labels=range(n_classes), normalize="true") for p in pred]
    return np.mean(cms, axis=0)

def main_score(y, proba):
    if proba.shape[2] == 2:
        return float(_auc_binary(y, proba[:, :, 1]).mean())
    return float(np.mean([_auc_binary((y == c).astype(int), proba[:, :, c]).mean()
                          for c in range(proba.shape[2])]))

def mean_roc(y, proba, grid=np.linspace(0, 1, 101)):
    tprs = []
    for p in proba:
        fpr, tpr, _ = roc_curve(y, p[:, 1])
        tprs.append(np.interp(grid, fpr, tpr))
    tpr = np.mean(tprs, axis=0)
    tpr[0] = 0.0
    return grid, tpr

def auc_per_repeat(y, proba):
    if proba.shape[2] == 2:
        return _auc_binary(y, proba[:, :, 1])
    return np.mean([_auc_binary((y == c).astype(int), proba[:, :, c])
                    for c in range(proba.shape[2])], axis=0)

def paired_difference(y, proba_a, proba_b, n_boot=N_BOOT, seed=0):
    d0 = float(auc_per_repeat(y, proba_b).mean() - auc_per_repeat(y, proba_a).mean())
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        b = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_class])
        boot[i] = (auc_per_repeat(y[b], proba_b[:, b]).mean()
                   - auc_per_repeat(y[b], proba_a[:, b]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return d0, float(lo), float(hi), float(p)

def outer_splits(y, n_repeats=N_REPEATS, seed=SEED):
    outer = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=n_repeats, random_state=seed)
    return [(f // N_FOLDS, te) for f, (_, te) in enumerate(outer.split(np.zeros((len(y), 1)), y))]

def foldwise_auc(y, proba, seed=SEED):
    vals = []
    for r, te in outer_splits(y, proba.shape[0], seed):
        vals.append(float(auc_per_repeat(y[te], proba[r:r + 1, te])[0]))
    return np.array(vals)
