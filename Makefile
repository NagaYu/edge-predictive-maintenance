# エッジAI 予防保守デモ — ワンコマンド実行
.PHONY: all install data train infer demo clean

all: demo

install:
	pip install -r requirements.txt

data:
	python3 data/generate_data.py

train:
	python3 src/train.py

infer:
	python3 src/edge_inference.py

# データ生成 → 学習 → 推論 を一気通貫で実行
demo: data train infer

clean:
	rm -f data/*.csv models/*.cbm models/*.json
	rm -rf src/__pycache__
