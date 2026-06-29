# Edge-AI predictive-maintenance demo — one-command run
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

# Run the whole pipeline end to end: generate data -> train -> infer
demo: data train infer

clean:
	rm -f data/*.csv models/*.cbm models/*.json
	rm -rf src/__pycache__
