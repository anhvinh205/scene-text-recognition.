.PHONY: install install-cpu lint format test data train-det train-rec eval serve app docker-build docker-run onnx

install:            ## full dev install
	pip install -e ".[all]"

install-cpu:        ## CPU-only torch (CI / laptops without GPU)
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[all]"

lint:
	ruff check src tests app
	ruff format --check src tests app

format:
	ruff check --fix src tests app
	ruff format src tests app

test:
	pytest --cov=scenetext --cov-report=term-missing

data:               ## expects data/SceneTrialTrain/ (unzip icdar2003.zip into data)
	scenetext prepare-data

train-det:
	scenetext train-det

train-rec:
	scenetext train-rec

eval:
	scenetext eval-det
	scenetext eval-rec
	scenetext eval-e2e

serve:
	scenetext serve

app:
	streamlit run app/streamlit_app.py

onnx:
	scenetext export-onnx

docker-build:
	docker build -t scene-text-recognition:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/runs:/app/runs:ro scene-text-recognition:latest
