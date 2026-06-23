"""
特徴量定義（学習バッチ・ストリーミング推論で共通）。

設計上の最重要ポイント — リーク(データ漏洩)の排除:
  すべてのローリング特徴量は「trailing（後方）ウィンドウ」= 現在を含む過去 WINDOW 点のみ
  を参照する因果的(causal)な統計量である。未来の値を一切参照しないため、
  全系列に対して一括計算しても「未来→過去」へのリークは構造的に発生しない。
  （fold 分割後の評価データ側の行は、その行自身の過去しか見ない）

候補特徴量はすべて O(window) で計算でき、過去 WINDOW 点だけ保持すればよいためエッジに適する。
ノイズ特徴量(noise_*)は「特徴量選択が無価値な特徴を自動除去できるか」を検証するために
学習時のみ混入させる（推論側では計算しない）。
"""

import numpy as np
import pandas as pd

WINDOW = 10  # ローリングウィンドウ幅（現在を含む過去10点）

BASE_COLS = ["sensor_vibration", "sensor_temperature"]

# 実特徴量（学習・推論で計算する本物の候補）。順序固定。
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

# 学習時のみ混入させるノイズ特徴量（特徴量選択で除去されるべきもの）
NOISE_FEATURES = ["noise_gauss", "noise_uniform"]

# 全候補（学習時の選択前）
CANDIDATE_FEATURES = REAL_FEATURES + NOISE_FEATURES


def build_features_df(df: pd.DataFrame, with_noise: bool = True,
                      seed: int = 0) -> pd.DataFrame:
    """学習用: 全候補特徴量を一括(ベクトル化)で算出する。

    pandas の rolling は trailing ウィンドウ（因果的）であり、未来を参照しない。
    """
    out = pd.DataFrame(index=df.index)
    out["sensor_vibration"] = df["sensor_vibration"].to_numpy()
    out["sensor_temperature"] = df["sensor_temperature"].to_numpy()

    for col, p in [("sensor_vibration", "vib"), ("sensor_temperature", "temp")]:
        s = df[col]
        roll = s.rolling(window=WINDOW, min_periods=1)
        roll_mean = roll.mean()
        roll_std = roll.std().fillna(0.0)          # 単一点は std=NaN → 0
        roll_max = roll.max()
        roll_min = roll.min()
        out[f"{p}_roll_mean"] = roll_mean.to_numpy()
        out[f"{p}_roll_std"] = roll_std.to_numpy()
        out[f"{p}_diff_mean"] = (s - roll_mean).to_numpy()
        out[f"{p}_roll_range"] = (roll_max - roll_min).to_numpy()
        out[f"{p}_rate"] = s.diff().fillna(0.0).to_numpy()  # 1階差分（現在-直前）

    if with_noise:
        rng = np.random.default_rng(seed)
        n = len(df)
        out["noise_gauss"] = rng.normal(0.0, 1.0, n)
        out["noise_uniform"] = rng.uniform(-1.0, 1.0, n)

    # 列順を固定
    cols = REAL_FEATURES + (NOISE_FEATURES if with_noise else [])
    return out[cols]


def compute_point_features(vib_buf, temp_buf) -> np.ndarray:
    """推論用: deque バッファ(過去 WINDOW 点)から実特徴量ベクトルを numpy で算出。

    build_features_df と完全に同じロジック（trailing 統計, std ddof=1, 初期点0埋め）。
    返り値の並びは REAL_FEATURES に一致する。
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
