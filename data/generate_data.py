"""
インフラ（鉄道・送電網）の振動・温度センサーを模したダミーデータ生成スクリプト。

- 正常運転：基準値まわりに緩やかな日内変動 + ノイズ
- 異常パターン:
    1) 予兆型 (gradual): 一定区間にかけて値がじわじわ上昇し、最後に異常としてラベル付け
    2) スパイク型 (spike): 突発的に値が跳ね上がる単発異常
- 異常率は全体の約1%に調整
- train / test の2ファイルを出力
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# センサーの正常運転時の基準値
VIB_BASE = 2.0       # 振動 (mm/s 相当)
VIB_NOISE = 0.15
TEMP_BASE = 45.0     # 温度 (degC 相当)
TEMP_NOISE = 0.8


def _base_signal(n: int, start_idx: int = 0) -> pd.DataFrame:
    """正常運転のベース信号（緩やかな日内変動 + ノイズ）を生成する。"""
    t = np.arange(start_idx, start_idx + n)
    # 1日 = 1440分周期の緩やかな変動を模す
    daily_vib = 0.25 * np.sin(2 * np.pi * t / 1440.0)
    daily_temp = 3.0 * np.sin(2 * np.pi * t / 1440.0 + 0.5)

    vib = VIB_BASE + daily_vib + RNG.normal(0, VIB_NOISE, n)
    temp = TEMP_BASE + daily_temp + RNG.normal(0, TEMP_NOISE, n)

    df = pd.DataFrame(
        {
            "sensor_vibration": vib,
            "sensor_temperature": temp,
            "is_anomaly": np.zeros(n, dtype=int),
        }
    )
    return df


def _inject_gradual(df: pd.DataFrame, center: int, length: int) -> None:
    """予兆型異常: center を中心に length 区間で値が上昇していく劣化を注入。

    異常ラベルは上昇がピークに達する終盤のみに付与する（予兆を学習させる狙い）。
    """
    n = len(df)
    start = max(0, center - length // 2)
    end = min(n, center + length // 2)
    span = end - start
    if span <= 0:
        return
    # 0 -> 1 へ単調増加するランプ
    ramp = np.linspace(0.0, 1.0, span)
    df.loc[start:end - 1, "sensor_vibration"] += ramp * 4.5   # 振動が増大
    df.loc[start:end - 1, "sensor_temperature"] += ramp * 18.0  # 温度も上昇
    # 終盤25%を異常としてラベル付け
    label_start = start + int(span * 0.75)
    df.loc[label_start:end - 1, "is_anomaly"] = 1


def _inject_spike(df: pd.DataFrame, idx: int) -> None:
    """スパイク型異常: 単発の突発スパイクを注入。"""
    df.loc[idx, "sensor_vibration"] += RNG.uniform(6.0, 9.0)
    df.loc[idx, "sensor_temperature"] += RNG.uniform(15.0, 25.0)
    df.loc[idx, "is_anomaly"] = 1


def generate(n: int, start_idx: int, seed_offset: int) -> pd.DataFrame:
    df = _base_signal(n, start_idx=start_idx)

    target_anom = int(n * 0.01)  # 異常率 約1%

    # 予兆型: 各イベントが終盤25%(=length*0.25)分のラベルを生む
    grad_length = 200                       # 1イベントの全長
    labels_per_grad = int(grad_length * 0.25)  # 約50ラベル
    n_grad_events = max(1, int(target_anom * 0.6) // labels_per_grad)
    n_grad_events = max(1, n_grad_events)

    grad_label_count = 0
    used_regions = []
    for _ in range(n_grad_events):
        center = int(RNG.integers(grad_length, n - grad_length))
        _inject_gradual(df, center, grad_length)
        used_regions.append((center - grad_length, center + grad_length))

    grad_label_count = int(df["is_anomaly"].sum())

    # スパイク型: 残りの異常枠を単発スパイクで埋める
    n_spikes = max(1, target_anom - grad_label_count)
    placed = 0
    attempts = 0
    while placed < n_spikes and attempts < n_spikes * 50:
        attempts += 1
        idx = int(RNG.integers(0, n))
        if df.loc[idx, "is_anomaly"] == 1:
            continue
        _inject_spike(df, idx)
        placed += 1

    return df


def main():
    # 時刻列を 1分間隔で付与
    train = generate(n=20000, start_idx=0, seed_offset=0)
    test = generate(n=8000, start_idx=20000, seed_offset=1)

    train_ts = pd.date_range("2025-01-01 00:00:00", periods=len(train), freq="min")
    test_ts = pd.date_range("2025-01-15 00:00:00", periods=len(test), freq="min")

    train.insert(0, "timestamp", train_ts)
    test.insert(0, "timestamp", test_ts)

    train = train[["timestamp", "sensor_vibration", "sensor_temperature", "is_anomaly"]]
    test = test[["timestamp", "sensor_vibration", "sensor_temperature", "is_anomaly"]]

    train.to_csv("data/train_sensor_data.csv", index=False)
    test.to_csv("data/test_sensor_data.csv", index=False)

    for name, df in [("train", train), ("test", test)]:
        n = len(df)
        n_anom = int(df["is_anomaly"].sum())
        print(f"[{name}] rows={n}, anomalies={n_anom}, anomaly_rate={n_anom / n * 100:.2f}%")
    print("Saved: data/train_sensor_data.csv, data/test_sensor_data.csv")


if __name__ == "__main__":
    main()
