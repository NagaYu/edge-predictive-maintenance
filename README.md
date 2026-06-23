# エッジAI 予防保守・異常検知パイプライン（PoC）

[English README →](README.en.md)

![status](https://img.shields.io/badge/status-proof--of--concept-orange)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

鉄道・送電網などのインフラ設備の振動／温度センサーを対象とした、**完全ローカル・低リソース**で動くエッジAIの参照実装。リークのない厳密なバリデーション、特徴量選択、ハイパーパラメータ最適化、低遅延・省メモリ推論までを、プロダクションで通用する“作法”で実装している。

> ⚠️ **これは合成データによる Proof of Concept です。**
> センサーデータは `data/generate_data.py` が決定論的に生成した**ダミーデータ**であり、
> 本 README に記載のスコア（F1≈0.89, PR-AUC≈0.98, recall=1.0）は理想化された値です。
> **実データでは必ず低下します。実機・本番運用にそのまま投入しないでください。**
> 目的は「手法とエンジニアリングの実証」であり、学習・技術検証・提案のための教材です。

## 構成

```
.
├── requirements.txt
├── data/
│   ├── generate_data.py            # ダミーデータ生成（予兆型+スパイク型, 異常率1%）
│   ├── train_sensor_data.csv       # 20,000行
│   └── test_sensor_data.csv        # 8,000行
├── src/
│   ├── features.py                 # 特徴量定義（学習バッチ・推論で共通 / 因果的=リークなし）
│   ├── train.py                    # CV検証 + 特徴量選択 + HPO + 軽量CatBoost
│   └── edge_inference.py           # deque低遅延推論 + プロファイリング
└── models/
    ├── edge_anomaly_model.cbm       # 学習済みモデル（ネイティブバイナリ, 約42KB）
    └── selected_features.json       # 採用特徴量・閾値・窓幅
```

## 実行

```bash
pip install -r requirements.txt
python3 data/generate_data.py
python3 src/train.py
python3 src/edge_inference.py

# もしくはワンコマンドで一気通貫
make demo
```

## ベンチマーク（合成データ・参考値）

| 指標 | 値 |
|---|---|
| F1-score (test) | 0.894 |
| PR-AUC (test) | 0.984 |
| Recall（異常） | 1.000（80/80 捕捉） |
| CV PR-AUC | 0.902 ± 0.196（TimeSeriesSplit 5-fold） |
| モデルサイズ | 42.5 KB（CatBoost ネイティブ） |
| 推論レイテンシ | 0.25 ms/行（deque ストリーミング） |
| 推論メモリ | 91 KB ピーク（tracemalloc） |

> CV の高分散（±0.196）は最終 fold での性能低下に由来する。これは TimeSeriesSplit が
> **時間的分布シフトを正しく顕在化**させた結果であり、シャッフルCVでは隠れてしまう。
> 「正直なリスク指標」として report に残している（隠さない）。

## 1. リークのない厳密なバリデーション
- **TimeSeriesSplit**（n_splits=5）：時系列の順序を壊さず、未来データで過去を評価しない。シャッフルCVが隠してしまう時間的な汎化劣化（本データでは最終fold）を顕在化させる。
- **前処理のリーク防止**：`StandardScaler` を各 fold の学習部分のみで `fit`。`leakfree_cv()` で手動徹底（CatBoost 早期終了を併用するため）し、さらに sklearn `Pipeline(StandardScaler→CatBoost)` + `cross_validate` でも同等のリークフリー検証を提示。
- **特徴量の因果性**：ローリング統計はすべて trailing（現在を含む過去 WINDOW 点）のみ参照するため、全系列一括計算でも未来→過去のリークが構造的に発生しない。

## 2. 特徴量選択とハイパーパラメータ最適化
- **特徴量選択**：候補14（実12＋ノイズ2）に対し CatBoost 重要度を算出し、寄与1%未満を自動除外。混入させたノイズ特徴量2つは確実に除去される（8特徴量を採用）。
- **HPO**：`depth` / `l2_leaf_reg` / `min_data_in_leaf` を TimeSeriesSplit CV（PR-AUC）で比較選定。`bagging_temperature`・`early_stopping_rounds` も設定。希少な正例では過正則化が再現率を破壊するため、分散とのトレードオフを CV で確認しながらチューニングする。
- **データを捨てない最終学習**：CV由来の堅牢な木本数で**全学習データ**を使い再学習し、ネイティブ `.cbm`（最小・高速ロード）で保存。

## 3. 低遅延・省メモリ推論
- **pandas 行ループ禁止**：ストリームは NumPy 配列から供給し、`collections.deque(maxlen=WINDOW)` のスライディングウィンドウ＋NumPy で特徴量を逐次計算。
- **ベクトル化バッチ**：蓄積データは NumPy 一括スコアリング（参考ベンチで約1,000万行/秒）。
- CatBoost はスケール不変のため最終成果物にスケーラを持たせず、推論を最小遅延化。
- `tracemalloc` と最大RSSでメモリを、`time.perf_counter` でms単位レイテンシ（mean/median/P95/P99）を計測。

## 制約と本番化への道のり（Limitations & Roadmap）

このPoCは**手法の実証**を目的としており、実インフラへ投入する前には以下が必須です。

- [ ] **実データでの検証** — 合成パターンではなく実センサーのドリフト・季節変動・実故障モードで再評価する。
- [ ] **時間的分布シフトへの対策** — CV最終foldの性能低下（PR-AUC 0.51）が示すとおり、未知の異常タイプへの頑健性を確保する（再学習トリガ、ドリフト検知）。
- [ ] **閾値のコスト設計** — 安全クリティカル系では固定0.5ではなく、見逃し（recall）と誤報（precision/アラート疲れ）のビジネス的トレードオフで決定する。
- [ ] **MLOps** — モデルのバージョン管理、監視、定期再学習、ロールバック、データ品質チェック。
- [ ] **エッジ実機での検証** — 目標ハードウェア（Raspberry Pi 等）での実レイテンシ・消費電力・長時間安定性。

> 鉄道・送電網は人命に関わる安全クリティカル系です。本リポジトリのコード・モデルを
> 実運用の保全判断に用いないでください（LICENSE の無保証条項を参照）。

## License

[MIT](LICENSE) — 教育・技術検証目的の参照実装として提供。
