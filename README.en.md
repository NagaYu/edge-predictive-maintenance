# Edge AI for Predictive Maintenance & Anomaly Detection (PoC)

[日本語版 README →](README.md)

![status](https://img.shields.io/badge/status-proof--of--concept-orange)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

A reference implementation of a **fully-local, low-resource** edge-AI pipeline for vibration/temperature sensors on large-scale infrastructure (railways, power grids). It demonstrates production-grade practice end to end: leak-free validation, feature selection, hyperparameter optimization, and low-latency / low-memory inference.

> ⚠️ **This is a synthetic-data Proof of Concept.**
> The sensor data is **dummy data** generated deterministically by `data/generate_data.py`,
> and the scores reported here (F1≈0.89, PR-AUC≈0.98, recall=1.0) are idealized.
> **They will degrade on real data. Do not deploy this as-is to real equipment or production.**
> The goal is to demonstrate the *method and engineering* — it is teaching/evaluation material.

## Layout

```
.
├── requirements.txt
├── Makefile
├── data/
│   ├── generate_data.py            # Synthetic data (gradual + spike anomalies, ~1% anomaly rate)
│   ├── train_sensor_data.csv       # 20,000 rows (generated)
│   └── test_sensor_data.csv        # 8,000 rows (generated)
├── src/
│   ├── features.py                 # Feature definitions, shared by training & inference (causal = leak-free)
│   ├── train.py                    # CV validation + feature selection + HPO + lightweight CatBoost
│   └── edge_inference.py           # Low-latency deque inference + profiling
└── models/
    ├── edge_anomaly_model.cbm       # Trained model (native binary, ~42 KB; generated)
    └── selected_features.json       # Selected features, threshold, window size (generated)
```

## Quickstart

```bash
pip install -r requirements.txt
python3 data/generate_data.py
python3 src/train.py
python3 src/edge_inference.py

# or run the whole pipeline in one command
make demo
```

## Benchmarks (synthetic data, indicative)

| Metric | Value |
|---|---|
| F1-score (test) | 0.894 |
| PR-AUC (test) | 0.984 |
| Recall (anomaly) | 1.000 (80/80 caught) |
| CV PR-AUC | 0.902 ± 0.196 (TimeSeriesSplit, 5-fold) |
| Model size | 42.5 KB (CatBoost native) |
| Inference latency | 0.25 ms/row (deque streaming) |
| Inference memory | 91 KB peak (tracemalloc) |

> The high CV variance (±0.196) comes from a performance drop on the final fold. This is
> **TimeSeriesSplit correctly surfacing temporal distribution shift** — something a shuffled CV
> would hide. We keep it in the report as an honest risk signal rather than hiding it.

## 1. Leak-free, rigorous validation
- **TimeSeriesSplit** (n_splits=5): preserves temporal order; never evaluates the past with future data. Surfaces the temporal generalization drop (final fold here) that shuffled CV would mask.
- **No preprocessing leakage**: `StandardScaler` is `fit` only on each fold's training portion. Enforced manually in `leakfree_cv()` (to allow CatBoost early stopping), and additionally demonstrated with a canonical sklearn `Pipeline(StandardScaler → CatBoost)` + `cross_validate`.
- **Causal features**: all rolling statistics use a trailing window (the current point plus the past WINDOW points only), so computing them over the full series introduces no future→past leakage.

## 2. Feature selection & hyperparameter optimization
- **Feature selection**: from 14 candidates (12 real + 2 injected noise), CatBoost importances drop anything contributing < 1%. Both injected noise features are reliably removed (8 features retained).
- **HPO**: `depth` / `l2_leaf_reg` / `min_data_in_leaf` are selected via TimeSeriesSplit CV (PR-AUC); `bagging_temperature` and `early_stopping_rounds` are set as well. Because over-regularization destroys recall when positives are rare, the variance/recall trade-off is verified with CV during tuning.
- **No wasted data**: the final model is retrained on the **full** training set using a robust CV-derived tree count, then saved as native `.cbm` (smallest size, fastest load).

## 3. Low-latency, low-memory inference
- **No pandas row loops**: the stream is fed from a NumPy array; features are computed incrementally with a `collections.deque(maxlen=WINDOW)` sliding window + NumPy.
- **Vectorized batch path**: buffered data is scored in a single NumPy pass (~10M rows/s in the reference benchmark).
- CatBoost is scale-invariant, so the deployed artifact carries no scaler — minimizing inference latency.
- Memory measured with `tracemalloc` and max RSS; latency (mean/median/P95/P99, in ms) measured with `time.perf_counter`.

## Limitations & Roadmap

This PoC demonstrates *method*. Before any real deployment, the following are mandatory:

- [ ] **Validate on real data** — re-evaluate on real sensor drift, seasonality, and actual failure modes, not synthetic patterns.
- [ ] **Handle temporal distribution shift** — the final-fold drop (PR-AUC 0.51) shows the need for robustness to unseen anomaly types (retraining triggers, drift detection).
- [ ] **Cost-based thresholding** — for safety-critical systems, set the threshold from the business trade-off between misses (recall) and false alarms (precision / alert fatigue), not a fixed 0.5.
- [ ] **MLOps** — model versioning, monitoring, scheduled retraining, rollback, data-quality checks.
- [ ] **On-device validation** — real latency, power draw, and long-run stability on the target hardware (e.g. Raspberry Pi).

> Railways and power grids are safety-critical, life-affecting systems. Do not use this code or
> model for real maintenance decisions (see the no-warranty clause in the LICENSE).

## License

[MIT](LICENSE) — provided as a reference implementation for education and technical evaluation.
