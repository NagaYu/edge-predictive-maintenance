"""
Generate dummy data emulating vibration/temperature sensors on infrastructure
equipment (railways, power grids).

- Normal operation: a gentle daily oscillation around a baseline + noise.
- Anomaly patterns:
    1) gradual: values slowly ramp up over a region; only the tail is labeled
       anomalous (so the model can learn the early "precursor" signature).
    2) spike: a sudden, single-point burst.
- The anomaly rate is tuned to ~1% of all rows.
- Outputs two files: train and test.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Baseline values during normal operation
VIB_BASE = 2.0       # vibration (≈ mm/s)
VIB_NOISE = 0.15
TEMP_BASE = 45.0     # temperature (≈ degC)
TEMP_NOISE = 0.8


def _base_signal(n: int, start_idx: int = 0) -> pd.DataFrame:
    """Generate the normal-operation base signal (gentle daily drift + noise)."""
    t = np.arange(start_idx, start_idx + n)
    # Emulate a gentle oscillation with a 1-day (1440-minute) period
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
    """Inject a gradual degradation centered at `center`, ramping over `length`.

    Only the final portion of the ramp is labeled anomalous (to teach the
    precursor signature rather than the fully-degraded state).
    """
    n = len(df)
    start = max(0, center - length // 2)
    end = min(n, center + length // 2)
    span = end - start
    if span <= 0:
        return
    # Monotonic ramp from 0 -> 1
    ramp = np.linspace(0.0, 1.0, span)
    df.loc[start:end - 1, "sensor_vibration"] += ramp * 4.5    # vibration grows
    df.loc[start:end - 1, "sensor_temperature"] += ramp * 18.0  # temperature rises too
    # Label the last 25% of the ramp as anomalous
    label_start = start + int(span * 0.75)
    df.loc[label_start:end - 1, "is_anomaly"] = 1


def _inject_spike(df: pd.DataFrame, idx: int) -> None:
    """Inject a single-point spike anomaly."""
    df.loc[idx, "sensor_vibration"] += RNG.uniform(6.0, 9.0)
    df.loc[idx, "sensor_temperature"] += RNG.uniform(15.0, 25.0)
    df.loc[idx, "is_anomaly"] = 1


def generate(n: int, start_idx: int, seed_offset: int) -> pd.DataFrame:
    df = _base_signal(n, start_idx=start_idx)

    target_anom = int(n * 0.01)  # anomaly rate ≈ 1%

    # Gradual: each event produces tail labels (= length * 0.25)
    grad_length = 200                          # full length of one event
    labels_per_grad = int(grad_length * 0.25)  # ≈ 50 labels
    n_grad_events = max(1, int(target_anom * 0.6) // labels_per_grad)
    n_grad_events = max(1, n_grad_events)

    grad_label_count = 0
    used_regions = []
    for _ in range(n_grad_events):
        center = int(RNG.integers(grad_length, n - grad_length))
        _inject_gradual(df, center, grad_length)
        used_regions.append((center - grad_length, center + grad_length))

    grad_label_count = int(df["is_anomaly"].sum())

    # Spike: fill the remaining anomaly budget with single-point spikes
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
    # Attach a timestamp column at 1-minute intervals
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
