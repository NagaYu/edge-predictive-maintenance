"""
ステップ3(本番品質版): 低遅延・省メモリのスタンドアロン推論 + プロファイリング。

満たす要件:
  [3] 推論速度とメモリの極限化
      - pandas の1行ずつループ(iterrows/itertuples)を排除。
      - ストリーミングは collections.deque のスライディングウィンドウ + NumPy で特徴量計算。
      - バッチ推論は完全ベクトル化(NumPy一括)。
      - モデルは CatBoost ネイティブ .cbm をロード（高速・小サイズ）。
  プロファイリング: 1行あたりレイテンシ(ms, mean/median/P95/P99)、スループット、
                    tracemalloc によるPythonピークメモリ、RSS(最大常駐メモリ)を計測・報告。

ネットワーク非接続のエッジ端末を想定し、推論ループ内に外部通信・ディスクI/Oは一切なし。
"""

import json
import os
import sys
import time
import tracemalloc
from collections import deque

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import compute_point_features, build_features_df, REAL_FEATURES, WINDOW

MODEL_PATH = os.path.join(ROOT, "models", "edge_anomaly_model.cbm")
FEATURES_PATH = os.path.join(ROOT, "models", "selected_features.json")
STREAM_CSV = os.path.join(ROOT, "data", "test_sensor_data.csv")


def _max_rss_kb() -> float:
    """プロセスの最大常駐メモリ(KB)。macOSはbytes, Linuxはkbで返るため吸収。"""
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS は bytes
        return ru / 1024.0
    return float(ru)  # Linux は KB


class SlidingWindowExtractor:
    """直近 WINDOW 点のみを deque で保持する省メモリ特徴量抽出器。"""

    __slots__ = ("vib", "temp", "sel_idx")

    def __init__(self, window: int, sel_idx: np.ndarray):
        self.vib = deque(maxlen=window)
        self.temp = deque(maxlen=window)
        self.sel_idx = sel_idx  # REAL_FEATURES のうち採用列のインデックス

    def push(self, vibration: float, temperature: float) -> np.ndarray:
        self.vib.append(vibration)
        self.temp.append(temperature)
        feats = compute_point_features(self.vib, self.temp)  # (12,) NumPy
        return feats[self.sel_idx]                            # 採用列だけに絞る


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("モデル未検出。先に src/train.py を実行してください。")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    selected = meta["selected_features"]
    threshold = meta.get("threshold", 0.5)
    # REAL_FEATURES 内での採用列インデックス（順序保持）
    sel_idx = np.array([REAL_FEATURES.index(c) for c in selected], dtype=np.int64)
    return model, selected, sel_idx, threshold


def run_streaming(model, sel_idx, threshold, stream_arr, truth_arr):
    """deque スライディングウィンドウによる1点ずつのリアルタイム推論。

    ストリームは NumPy 配列から供給（pandas行ループは不使用）。
    """
    extractor = SlidingWindowExtractor(WINDOW, sel_idx)
    latencies = np.empty(len(stream_arr), dtype=np.float64)
    n_alerts = n_caught = n_true = 0

    for i in range(stream_arr.shape[0]):
        vib = stream_arr[i, 0]
        temp = stream_arr[i, 1]
        truth = truth_arr[i]

        t0 = time.perf_counter()
        feat = extractor.push(vib, temp)
        proba = model.predict_proba(feat)[1]      # 単一サンプル推論
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        if truth == 1:
            n_true += 1
        if proba >= threshold:
            n_alerts += 1
            if truth == 1:
                n_caught += 1
            print(f"[ALERT] 異常の予兆を検知しました。予測確率: {proba * 100:.1f}%"
                  f"  (idx={i}, vib={vib:.2f}, temp={temp:.1f})")

    return latencies, n_alerts, n_true, n_caught


def run_vectorized(model, selected, df):
    """参考: 完全ベクトル化バッチ推論（蓄積済みデータの一括スコアリング）。"""
    feats = build_features_df(df, with_noise=False)[selected]
    X = feats.to_numpy()
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return proba, elapsed_ms, len(X)


def main():
    model, selected, sel_idx, threshold = load_artifacts()

    # ストリーム源を一度だけ NumPy 化（実機ではセンサードライバ相当）
    df = pd.read_csv(STREAM_CSV)
    stream_arr = df[["sensor_vibration", "sensor_temperature"]].to_numpy(np.float64)
    truth_arr = df["is_anomaly"].to_numpy(np.int64)

    print("=" * 64)
    print(" エッジ・リアルタイム推論 (deque スライディングウィンドウ / 完全ローカル)")
    print("=" * 64)
    print(f" モデル: {os.path.basename(MODEL_PATH)}  採用特徴量: {len(selected)}  バッファ長: {WINDOW}")
    print("-" * 64)

    # メモリ計測開始（推論全体のPythonアロケーションを追跡）
    tracemalloc.start()

    latencies, n_alerts, n_true, n_caught = run_streaming(
        model, sel_idx, threshold, stream_arr, truth_arr
    )

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 参考: ベクトル化バッチ推論
    _, batch_ms, n_batch = run_vectorized(model, selected, df)

    print("-" * 64)
    print(" プロファイリング結果")
    print("-" * 64)
    print(f" 処理行数            : {len(stream_arr):,}")
    print(f" アラート発報 / 捕捉 : {n_alerts} / {n_caught}  "
          f"(true={n_true}, recall={n_caught / max(n_true, 1):.2%})")
    print("  [ストリーミング(deque, 1点ずつ)]")
    print(f"   平均レイテンシ     : {latencies.mean():.4f} ms / 行")
    print(f"   中央値             : {np.median(latencies):.4f} ms / 行")
    print(f"   P95 / P99          : {np.percentile(latencies,95):.4f} / {np.percentile(latencies,99):.4f} ms")
    print(f"   スループット       : {1000.0 / latencies.mean():,.0f} 行/秒")
    print("  [ベクトル化バッチ(NumPy一括)]")
    print(f"   {n_batch:,}行を一括 : {batch_ms:.3f} ms  "
          f"({batch_ms / n_batch * 1000:.4f} µs/行, {n_batch / (batch_ms/1000):,.0f} 行/秒)")
    print("  [メモリ]")
    print(f"   tracemalloc 現在/ピーク : {cur_mem/1024:.1f} / {peak_mem/1024:.1f} KB (推論中のPythonアロケーション)")
    print(f"   プロセス最大RSS        : {_max_rss_kb()/1024:.1f} MB")
    print(f"   モデルファイルサイズ   : {os.path.getsize(MODEL_PATH)/1024:.2f} KB")
    print("=" * 64)


if __name__ == "__main__":
    main()
