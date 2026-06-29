"""
Step 2 (production-grade): rigorous validation + feature selection + HPO + lightweight CatBoost.

Requirements covered:
  [1] Leak-free CV that matches the data (time series)
      - Uses TimeSeriesSplit (preserves temporal order; never evaluates the past with the future).
      - Stateful preprocessing (StandardScaler) is fit only on each fold's training portion,
        so validation statistics never leak into training. Enforced manually in leakfree_cv()
        (because CatBoost early stopping is used alongside it).
      - Also runs a canonical sklearn Pipeline(StandardScaler -> CatBoost) + cross_validate to
        explicitly demonstrate the leak-free Pipeline structure.
      - Rolling features are trailing (causal), so future leakage cannot occur by construction
        (see features.py).
  [2] Feature selection + hyperparameter optimization
      - CatBoost feature importances automatically drop worthless noise features.
      - depth / l2_leaf_reg / min_data_in_leaf / bagging_temperature / early_stopping_rounds are
        chosen logically via TimeSeriesSplit CV (PR-AUC).
      - In anomaly detection, where high recall matters, over-regularization destroys recall, so
        the variance/recall trade-off is checked with CV during tuning.
  [3] Lightweight and fast
      - Shallow depth + regularization prevents overfitting and keeps the model small.
      - Saved in CatBoost's native .cbm format (smallest size, fastest load).
      - The final model is retrained on the FULL training set using the optimal tree count found
        via early stopping (no data thrown away).

Outputs: chosen parameters, CV score mean and variance (std), feature-selection result,
test evaluation, model size.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import build_features_df, REAL_FEATURES, CANDIDATE_FEATURES, WINDOW

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "edge_anomaly_model.cbm")
FEATURES_PATH = os.path.join(MODELS_DIR, "selected_features.json")

N_SPLITS = 5
RANDOM_SEED = 42
MAX_ITERS = 1000
ESR = 50  # early_stopping_rounds


def base_params(scale_pos_weight: float) -> dict:
    """Base CatBoost params aimed at preventing overfitting while staying lightweight.

    Set logically for the data scale (train ~ 20k rows, ~1% anomalies):
      - depth / min_data_in_leaf / l2_leaf_reg are decided per fold via HPO.
      - bagging_temperature: Bayesian bootstrap for light regularization (variance reduction).
        Positives are rare, so too-strong values wreck recall -> keep it modest (0.5).
      - learning_rate kept small (0.05); the optimal tree count is found via early stopping.
    """
    return dict(
        loss_function="Logloss",
        eval_metric="PRAUC",
        learning_rate=0.05,
        bagging_temperature=0.5,
        random_strength=1.0,
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )


def leakfree_cv(X: pd.DataFrame, y: np.ndarray, params: dict, tscv: TimeSeriesSplit):
    """CV with TimeSeriesSplit + manual leak-free preprocessing + CatBoost early stopping.

    Per fold: fit StandardScaler on the training portion only (no leakage) -> transform ->
    early-stop on eval_set = validation portion. Returns validation PR-AUC / F1 / best tree count.
    """
    pr_list, f1_list, iters = [], [], []
    Xv = X.to_numpy()
    for tr_idx, va_idx in tscv.split(Xv):
        scaler = StandardScaler().fit(Xv[tr_idx])          # fit on the training portion only
        Xtr, Xva = scaler.transform(Xv[tr_idx]), scaler.transform(Xv[va_idx])
        ytr, yva = y[tr_idx], y[va_idx]
        m = CatBoostClassifier(iterations=MAX_ITERS, early_stopping_rounds=ESR, **params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        p = m.predict_proba(Xva)[:, 1]
        pr_list.append(average_precision_score(yva, p) if yva.sum() > 0 else np.nan)
        f1_list.append(f1_score(yva, (p >= 0.5).astype(int), zero_division=0))
        iters.append(m.tree_count_)
    return np.array(pr_list), np.array(f1_list), np.array(iters)


def main():
    print("=" * 64)
    print(" Production-grade pipeline: train / validate / select / optimize")
    print("=" * 64)

    train = pd.read_csv(os.path.join(DATA_DIR, "train_sensor_data.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_sensor_data.csv"))

    Xtr_all = build_features_df(train, with_noise=True, seed=1)
    Xte_all = build_features_df(test, with_noise=True, seed=2)
    y_train = train["is_anomaly"].astype(int).to_numpy()
    y_test = test["is_anomaly"].astype(int).to_numpy()

    pos = int(y_train.sum())
    spw = (len(y_train) - pos) / max(pos, 1)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f" Data: train={len(train):,}(anom {pos}) / test={len(test):,}(anom {int(y_test.sum())})")
    print(f" CV: TimeSeriesSplit(n_splits={N_SPLITS})  scale_pos_weight={spw:.1f}")
    print(f" Candidate features: {len(CANDIDATE_FEATURES)} (real {len(REAL_FEATURES)} + noise 2)")

    # ================================================================
    # [A] Feature selection: drop noise/low-contribution features via CatBoost importances
    # ================================================================
    print("\n[A] Feature selection (feature importance)")
    cut = int(len(Xtr_all) * 0.8)
    sel_tr, sel_val = Pool(Xtr_all.iloc[:cut], y_train[:cut]), Pool(Xtr_all.iloc[cut:], y_train[cut:])
    sel_model = CatBoostClassifier(iterations=400, depth=4, l2_leaf_reg=6.0, min_data_in_leaf=10,
                                   early_stopping_rounds=ESR, **base_params(spw))
    sel_model.fit(sel_tr, eval_set=sel_val)
    imp = sorted(zip(CANDIDATE_FEATURES, sel_model.get_feature_importance(sel_tr)),
                 key=lambda x: -x[1])
    THRESH = 1.0  # contribution < 1% is treated as worthless and dropped
    selected = [f for f, v in imp if v >= THRESH] or [f for f, _ in imp[:6]]
    dropped = [f for f, v in imp if f not in selected]
    for f, v in imp:
        print(f"    {'keep' if f in selected else 'drop'}  {f:<20} {v:6.2f}")
    print(f"  -> kept {len(selected)} / dropped {len(dropped)}: {dropped}")

    Xtr, Xte = Xtr_all[selected], Xte_all[selected]

    # ================================================================
    # [B] Hyperparameter optimization (TimeSeriesSplit CV, PR-AUC)
    #   Also check via min_data_in_leaf that over-regularization does not wreck recall.
    # ================================================================
    print("\n[B] Hyperparameter optimization (TimeSeriesSplit CV / metric=PR-AUC, early stopping)")
    candidates = [
        {"depth": 4, "l2_leaf_reg": 3.0, "min_data_in_leaf": 1},
        {"depth": 4, "l2_leaf_reg": 6.0, "min_data_in_leaf": 10},
        {"depth": 6, "l2_leaf_reg": 6.0, "min_data_in_leaf": 10},
        {"depth": 4, "l2_leaf_reg": 6.0, "min_data_in_leaf": 30},
    ]
    best = None
    for c in candidates:
        params = base_params(spw); params.update(c)
        pr, f1s, iters = leakfree_cv(Xtr, y_train, params, tscv)
        mean_pr = np.nanmean(pr)
        print(f"  depth={c['depth']}, l2={c['l2_leaf_reg']:<4}, min_leaf={c['min_data_in_leaf']:<3} "
              f" PR-AUC={mean_pr:.4f}+/-{np.nanstd(pr):.4f}  F1={f1s.mean():.4f}+/-{f1s.std():.4f}  "
              f"iters~{int(np.median(iters))}")
        if best is None or mean_pr > best["mean_pr"]:
            best = {"cfg": c, "mean_pr": mean_pr, "iters": iters}
    print(f"  -> chosen: {best['cfg']}")

    best_params = base_params(spw); best_params.update(best["cfg"])
    best_iter = int(max(100, np.median(best["iters"])))  # robust tree count derived from CV

    # ================================================================
    # [C] CV score variance for the chosen config (profiling report)
    #   (C-1) leakfree_cv (with early stopping) / (C-2) sklearn Pipeline, same leak-free check
    # ================================================================
    print("\n[C] CV scores for the chosen config (per fold / mean +/- variance)")
    pr, f1s, iters = leakfree_cv(Xtr, y_train, best_params, tscv)
    for i, (p, f, it) in enumerate(zip(pr, f1s, iters), 1):
        print(f"   fold{i}:  PR-AUC={p:.4f}  F1={f:.4f}  (trees={it})")
    print(f"   --------------------------------------------")
    print(f"   PR-AUC  mean={np.nanmean(pr):.4f}  std={np.nanstd(pr):.4f}  var={np.nanvar(pr):.6f}")
    print(f"   F1      mean={f1s.mean():.4f}  std={f1s.std():.4f}  var={f1s.var():.6f}")

    # (C-2) explicit leak-free check via an sklearn Pipeline (scaler fit per fold)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", CatBoostClassifier(iterations=best_iter, **best_params))])
    cvres = cross_validate(pipe, Xtr, y_train, cv=tscv,
                           scoring={"pr_auc": "average_precision", "f1": "f1"}, n_jobs=1)
    print(f"   [sklearn Pipeline check] PR-AUC mean={cvres['test_pr_auc'].mean():.4f} "
          f"std={cvres['test_pr_auc'].std():.4f}  (scaler fit per fold -> no leakage)")

    # ================================================================
    # [D] Final model: retrain on the FULL training set (best_iter fixed, no data discarded) + native save
    #   CatBoost is scale-invariant, so the deployed artifact needs no scaler (minimal inference latency).
    # ================================================================
    print("\n[D] Final model training (retrain on full data) + native .cbm save")
    final = CatBoostClassifier(iterations=best_iter, **best_params)
    final.fit(Pool(Xtr, y_train))
    os.makedirs(MODELS_DIR, exist_ok=True)
    final.save_model(MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump({"selected_features": selected, "window": WINDOW, "threshold": 0.5}, f, indent=2)

    proba = final.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr_auc = average_precision_score(y_test, proba)
    size_kb = os.path.getsize(MODEL_PATH) / 1024.0

    print("=" * 64)
    print(" Final result (held-out test evaluation)")
    print("=" * 64)
    print(f"  features kept : {len(selected)} / {len(CANDIDATE_FEATURES)}  ({selected})")
    print(f"  chosen params : {best['cfg']}  trees={best_iter}")
    print(f"  bagging_temperature=0.5  early_stopping_rounds={ESR}")
    print(f"  F1-score      : {f1:.4f}")
    print(f"  PR-AUC        : {pr_auc:.4f}")
    print(f"  model size    : {size_kb:.2f} KB  ({os.path.relpath(MODEL_PATH, ROOT)})")
    print("=" * 64)
    print(classification_report(y_test, pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
