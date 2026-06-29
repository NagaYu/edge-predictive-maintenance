"""
Feature definitions, shared by both batch training and streaming inference.

The most important design point — eliminating leakage:
  Every rolling feature uses a *trailing* window = only the current point plus the
  past WINDOW points. It never looks at future values, so computing the features
  over the full series introduces no future->past leakage by construction.
  (After a fold split, a validation row only ever sees its own past.)

All candidate features are O(window) to compute and require keeping only the last
WINDOW points, which makes them suitable for the edge. The noise features (noise_*)
are injected at training time only, to verify that feature selection can remove
worthless features automatically (the inference path never computes them).
"""

import numpy as np
import pandas as pd

WINDOW = 10  # rolling window size (current point plus the past 10)

BASE_COLS = ["sensor_vibration", "sensor_temperature"]

# Real features (the genuine candidates computed in both training and inference). Fixed order.
REAL_FEATURES = [
    "sensor_vibration",
    "vib_roll_mean",
    "vib_roll_std",
    "vib_diff_mean",
    "vib_roll_range",
    "vib_rate",
    "sensor_temperature",
    "temp_roll_mean",
    "temp_roll_std",
    "temp_diff_mean",
    "temp_roll_range",
    "temp_rate",
]

# Noise features injected at training time only (should be removed by feature selection)
NOISE_FEATURES = ["noise_gauss", "noise_uniform"]

# All candidates (before selection, at training time)
CANDIDATE_FEATURES = REAL_FEATURES + NOISE_FEATURES


def build_features_df(df: pd.DataFrame, with_noise: bool = True,
                      seed: int = 0) -> pd.DataFrame:
    """Training path: compute all candidate features at once (vectorized).

    pandas' rolling uses a trailing (causal) window and never looks at the future.
    """
    out = pd.DataFrame(index=df.index)
    out["sensor_vibration"] = df["sensor_vibration"].to_numpy()
    out["sensor_temperature"] = df["sensor_temperature"].to_numpy()

    for col, p in [("sensor_vibration", "vib"), ("sensor_temperature", "temp")]:
        s = df[col]
        roll = s.rolling(window=WINDOW, min_periods=1)
        roll_mean = roll.mean()
        roll_std = roll.std().fillna(0.0)          # std of a single point is NaN -> 0
        roll_max = roll.max()
        roll_min = roll.min()
        out[f"{p}_roll_mean"] = roll_mean.to_numpy()
        out[f"{p}_roll_std"] = roll_std.to_numpy()
        out[f"{p}_diff_mean"] = (s - roll_mean).to_numpy()
        out[f"{p}_roll_range"] = (roll_max - roll_min).to_numpy()
        out[f"{p}_rate"] = s.diff().fillna(0.0).to_numpy()  # first difference (current - previous)

    if with_noise:
        rng = np.random.default_rng(seed)
        n = len(df)
        out["noise_gauss"] = rng.normal(0.0, 1.0, n)
        out["noise_uniform"] = rng.uniform(-1.0, 1.0, n)

    # Fix the column order
    cols = REAL_FEATURES + (NOISE_FEATURES if with_noise else [])
    return out[cols]


def compute_point_features(vib_buf, temp_buf) -> np.ndarray:
    """Inference path: compute the real feature vector from deque buffers (past WINDOW
    points) using NumPy.

    Exactly the same logic as build_features_df (trailing stats, std ddof=1, zero-fill
    on the first point). The output order matches REAL_FEATURES.
    """
    v = np.fromiter(vib_buf, dtype=np.float64)
    t = np.fromiter(temp_buf, dtype=np.float64)

    def stats(arr):
        cur = arr[-1]
        mean = arr.mean()
        std = arr.std(ddof=1) if arr.size > 1 else 0.0
        rng_ = arr.max() - arr.min()
        rate = (arr[-1] - arr[-2]) if arr.size > 1 else 0.0
        diff_mean = cur - mean
        return cur, mean, std, diff_mean, rng_, rate

    v_cur, v_mean, v_std, v_diff, v_rng, v_rate = stats(v)
    t_cur, t_mean, t_std, t_diff, t_rng, t_rate = stats(t)

    return np.array(
        [
            v_cur, v_mean, v_std, v_diff, v_rng, v_rate,
            t_cur, t_mean, t_std, t_diff, t_rng, t_rate,
        ],
        dtype=np.float64,
    )
