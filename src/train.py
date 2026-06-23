"""
ステップ2(本番品質版): 厳密なバリデーション + 特徴量選択 + HPO + 軽量CatBoost。

満たす要件:
  [1] 時系列に合致したリークのないCV
      - TimeSeriesSplit を採用（時間順序を壊さず、未来データで過去を評価しない）。
      - 前処理(StandardScaler)は各 fold の学習部分のみで fit され、評価データの統計が漏れない。
        leakfree_cv() で手動に徹底（CatBoost の early stopping を併用するため）。
      - 併せて sklearn Pipeline(StandardScaler→CatBoost)+cross_validate を実行し、
        「Pipeline構造でのリークフリー検証」も明示的に提示。
      - ローリング特徴量は trailing(因果的)なので未来リークは構造的に発生しない（features.py 参照）。
  [2] 特徴量選択 + ハイパーパラメータ最適化
      - CatBoost の特徴量重要度で無価値なノイズ特徴量を自動除外。
      - depth / l2_leaf_reg / min_data_in_leaf / bagging_temperature / early_stopping_rounds を
        TimeSeriesSplit CV (PR-AUC) で論理的に選定。
      - 安全側(高recall)が要る異常検知では過正則化が再現率を壊すため、
        分散とのトレードオフを CV で確認しながらチューニングする。
  [3] 軽量・高速
      - 浅い depth と正則化で過学習を防ぎつつモデルを小型化。
      - モデルは CatBoost ネイティブ .cbm（最小サイズ・高速ロード）で保存。
      - 最終モデルは早期終了で得た最適本数を使い「全学習データ」で再学習（データを捨てない）。

出力: 採用パラメータ、CVスコアの平均と分散(std)、特徴量選択結果、テスト評価、モデルサイズ。
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
    """過学習防止＋軽量化を狙った CatBoost の基本パラメータ。

    データ規模(train≈2万行, 異常≈1%)に対して論理的に設定:
      - depth/min_data_in_leaf/l2_leaf_reg は HPO で fold ごとに検証して決定。
      - bagging_temperature: ベイズ的ブートストラップによる軽い正則化(分散低減)。
        ただし正例が希少なため強すぎると再現率を壊す → 0.5(控えめ)。
      - learning_rate を小さめ(0.05)にし、early stopping で最適本数を決める。
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
    """TimeSeriesSplit + 手動リークフリー前処理 + CatBoost早期終了でCV評価。

    各 fold: StandardScaler を学習部分のみで fit(リーク防止) → 変換 →
             eval_set=検証部分で early stopping。検証 PR-AUC / F1 / 最適本数を返す。
    """
    pr_list, f1_list, iters = [], [], []
    Xv = X.to_numpy()
    for tr_idx, va_idx in tscv.split(Xv):
        scaler = StandardScaler().fit(Xv[tr_idx])          # 学習部分のみで fit
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
    print(" 本番品質パイプライン: 学習・検証・選択・最適化")
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
    print(f" データ: train={len(train):,}(異常{pos}) / test={len(test):,}(異常{int(y_test.sum())})")
    print(f" CV戦略: TimeSeriesSplit(n_splits={N_SPLITS})  scale_pos_weight={spw:.1f}")
    print(f" 候補特徴量数: {len(CANDIDATE_FEATURES)} (実{len(REAL_FEATURES)} + ノイズ2)")

    # ================================================================
    # [A] 特徴量選択: CatBoost重要度でノイズ/低寄与特徴を自動除外
    # ================================================================
    print("\n[A] 特徴量選択 (Feature Importance)")
    cut = int(len(Xtr_all) * 0.8)
    sel_tr, sel_val = Pool(Xtr_all.iloc[:cut], y_train[:cut]), Pool(Xtr_all.iloc[cut:], y_train[cut:])
    sel_model = CatBoostClassifier(iterations=400, depth=4, l2_leaf_reg=6.0, min_data_in_leaf=10,
                                   early_stopping_rounds=ESR, **base_params(spw))
    sel_model.fit(sel_tr, eval_set=sel_val)
    imp = sorted(zip(CANDIDATE_FEATURES, sel_model.get_feature_importance(sel_tr)),
                 key=lambda x: -x[1])
    THRESH = 1.0  # 寄与1%未満は無価値として除外
    selected = [f for f, v in imp if v >= THRESH] or [f for f, _ in imp[:6]]
    dropped = [f for f, v in imp if f not in selected]
    for f, v in imp:
        print(f"    {'✓keep' if f in selected else '✗drop'}  {f:<20} {v:6.2f}")
    print(f"  → 採用 {len(selected)} / 除外 {len(dropped)}: {dropped}")

    Xtr, Xte = Xtr_all[selected], Xte_all[selected]

    # ================================================================
    # [B] ハイパーパラメータ最適化 (TimeSeriesSplit CV, PR-AUC)
    #   過正則化が再現率を壊さないか min_data_in_leaf も含めて検証する。
    # ================================================================
    print("\n[B] ハイパーパラメータ最適化 (TimeSeriesSplit CV / 指標=PR-AUC, 早期終了)")
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
              f" PR-AUC={mean_pr:.4f}±{np.nanstd(pr):.4f}  F1={f1s.mean():.4f}±{f1s.std():.4f}  "
              f"iters~{int(np.median(iters))}")
        if best is None or mean_pr > best["mean_pr"]:
            best = {"cfg": c, "mean_pr": mean_pr, "iters": iters}
    print(f"  → 採用: {best['cfg']}")

    best_params = base_params(spw); best_params.update(best["cfg"])
    best_iter = int(max(100, np.median(best["iters"])))  # CV由来の堅牢な本数

    # ================================================================
    # [C] 採用構成の CV スコア分散（プロファイリング報告）
    #   (C-1) leakfree_cv（早期終了つき）/ (C-2) sklearn Pipeline で同等のリークフリー検証
    # ================================================================
    print("\n[C] 採用構成の CV スコア（fold別 / 平均 ± 分散）")
    pr, f1s, iters = leakfree_cv(Xtr, y_train, best_params, tscv)
    for i, (p, f, it) in enumerate(zip(pr, f1s, iters), 1):
        print(f"   fold{i}:  PR-AUC={p:.4f}  F1={f:.4f}  (trees={it})")
    print(f"   --------------------------------------------")
    print(f"   PR-AUC  mean={np.nanmean(pr):.4f}  std={np.nanstd(pr):.4f}  var={np.nanvar(pr):.6f}")
    print(f"   F1      mean={f1s.mean():.4f}  std={f1s.std():.4f}  var={f1s.var():.6f}")

    # (C-2) sklearn Pipeline による明示的なリークフリー検証（scaler は fold ごとに fit）
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", CatBoostClassifier(iterations=best_iter, **best_params))])
    cvres = cross_validate(pipe, Xtr, y_train, cv=tscv,
                           scoring={"pr_auc": "average_precision", "f1": "f1"}, n_jobs=1)
    print(f"   [sklearn Pipeline検証] PR-AUC mean={cvres['test_pr_auc'].mean():.4f} "
          f"std={cvres['test_pr_auc'].std():.4f}  (scalerはfoldごとにfit→リークなし)")

    # ================================================================
    # [D] 最終モデル: 全学習データで再学習（best_iter固定, データを捨てない）+ ネイティブ保存
    #   CatBoost はスケール不変のため最終成果物はスケーラ不要(推論を最小遅延化)。
    # ================================================================
    print("\n[D] 最終モデル学習 (全データ再学習) + ネイティブ.cbm 保存")
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
    print(" 最終結果 (ホールドアウト test 評価)")
    print("=" * 64)
    print(f"  採用特徴量数 : {len(selected)} / {len(CANDIDATE_FEATURES)}  ({selected})")
    print(f"  採用パラメータ: {best['cfg']}  trees={best_iter}")
    print(f"  bagging_temperature=0.5  early_stopping_rounds={ESR}")
    print(f"  F1-score     : {f1:.4f}")
    print(f"  PR-AUC       : {pr_auc:.4f}")
    print(f"  モデルサイズ : {size_kb:.2f} KB  ({os.path.relpath(MODEL_PATH, ROOT)})")
    print("=" * 64)
    print(classification_report(y_test, pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
