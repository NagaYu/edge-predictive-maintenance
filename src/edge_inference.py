"""
Step 3 (production-grade): low-latency, low-memory standalone inference + profiling.

Requirements covered:
  [3] Extreme inference speed and memory
      - No pandas row-by-row loop (iterrows/itertuples).
      - Streaming uses a collections.deque sliding window + NumPy for feature computation.
      - Batch inference is fully vectorized (single NumPy pass).
      - Loads the CatBoost native .cbm model (fast, small).
  Profiling: per-row latency (ms, mean/median/P95/P99), throughput, peak Python memory via
             tracemalloc, and max RSS (resident set size).

Targets a network-disconnected edge device: there is no external I/O inside the inference loop.
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
    """Process max RSS in KB. macOS reports bytes, Linux reports KB; normalize."""
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS: bytes
        return ru / 1024.0
    return float(ru)  # Linux: KB


class SlidingWindowExtractor:
    """Low-memory feature extractor that keeps only the last WINDOW points in a deque."""

    __slots__ = ("vib", "temp", "sel_idx")

    def __init__(self, window: int, sel_idx: np.ndarray):
        self.vib = deque(maxlen=window)
        self.temp = deque(maxlen=window)
        self.sel_idx = sel_idx  # indices of the selected columns within REAL_FEATURES

    def push(self, vibration: float, temperature: float) -> np.ndarray:
        self.vib.append(vibration)
        self.temp.append(temperature)
        feats = compute_point_features(self.vib, self.temp)  # (12,) NumPy
        return feats[self.sel_idx]                            # narrow to the selected columns


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found. Run src/train.py first.")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    selected = meta["selected_features"]
    threshold = meta.get("threshold", 0.5)
    # indices of the selected columns within REAL_FEATURES (order preserved)
    sel_idx = np.array([REAL_FEATURES.index(c) for c in selected], dtype=np.int64)
    return model, selected, sel_idx, threshold


def run_streaming(model, sel_idx, threshold, stream_arr, truth_arr):
    """Real-time, one-point-at-a-time inference via a deque sliding window.

    The stream is fed from a NumPy array (no pandas row loop).
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
        proba = model.predict_proba(feat)[1]      # single-sample inference
        latencies[i] = (time.perf_counter() - t0) * 1000.0

        if truth == 1:
            n_true += 1
        if proba >= threshold:
            n_alerts += 1
            if truth == 1:
                n_caught += 1
            print(f"[ALERT] Anomaly precursor detected. predicted probability: {proba * 100:.1f}%"
                  f"  (idx={i}, vib={vib:.2f}, temp={temp:.1f})")

    return latencies, n_alerts, n_true, n_caught


def run_vectorized(model, selected, df):
    """Reference: fully vectorized batch inference (one-shot scoring of buffered data)."""
    feats = build_features_df(df, with_noise=False)[selected]
    X = feats.to_numpy()
    t0 = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return proba, elapsed_ms, len(X)


def main():
    model, selected, sel_idx, threshold = load_artifacts()

    # Materialize the stream source as NumPy once (a sensor driver in production)
    df = pd.read_csv(STREAM_CSV)
    stream_arr = df[["sensor_vibration", "sensor_temperature"]].to_numpy(np.float64)
    truth_arr = df["is_anomaly"].to_numpy(np.int64)

    print("=" * 64)
    print(" Edge real-time inference (deque sliding window / fully local)")
    print("=" * 64)
    print(f" model: {os.path.basename(MODEL_PATH)}  features: {len(selected)}  buffer: {WINDOW}")
    print("-" * 64)

    # Start memory tracking (tracks Python allocations across the whole inference)
    tracemalloc.start()

    latencies, n_alerts, n_true, n_caught = run_streaming(
        model, sel_idx, threshold, stream_arr, truth_arr
    )

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Reference: vectorized batch inference
    _, batch_ms, n_batch = run_vectorized(model, selected, df)

    print("-" * 64)
    print(" Profiling results")
    print("-" * 64)
    print(f" rows processed      : {len(stream_arr):,}")
    print(f" alerts / caught     : {n_alerts} / {n_caught}  "
          f"(true={n_true}, recall={n_caught / max(n_true, 1):.2%})")
    print("  [streaming (deque, one point at a time)]")
    print(f"   mean latency       : {latencies.mean():.4f} ms / row")
    print(f"   median             : {np.median(latencies):.4f} ms / row")
    print(f"   P95 / P99          : {np.percentile(latencies,95):.4f} / {np.percentile(latencies,99):.4f} ms")
    print(f"   throughput         : {1000.0 / latencies.mean():,.0f} rows/sec")
    print("  [vectorized batch (single NumPy pass)]")
    print(f"   {n_batch:,} rows at once : {batch_ms:.3f} ms  "
          f"({batch_ms / n_batch * 1000:.4f} us/row, {n_batch / (batch_ms/1000):,.0f} rows/sec)")
    print("  [memory]")
    print(f"   tracemalloc current/peak : {cur_mem/1024:.1f} / {peak_mem/1024:.1f} KB (Python allocations during inference)")
    print(f"   process max RSS          : {_max_rss_kb()/1024:.1f} MB")
    print(f"   model file size          : {os.path.getsize(MODEL_PATH)/1024:.2f} KB")
    print("=" * 64)


if __name__ == "__main__":
    main()
