# Scene Text Recognition: YOLO26 + CRNN

[![CI](https://github.com/anhvinh205/scene-text-recognition./actions/workflows/ci.yml/badge.svg)](https://github.com/anhvinh205/scene-text-recognition./actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end scene text recognition: YOLO26 detects word regions and a CRNN (ResNet + BiLSTM + CTC)
recognizes the text in each crop. The project includes data preparation, training, evaluation,
a FastAPI service, a Streamlit demo, ONNX export, Docker support, and tests.

```text
image -> YOLO26 text detector -> word crops -> CRNN recognizer -> text and bounding boxes
```

> This repository is named `scene-text-recognition.` on GitHub (including the final period). Its
> HTTPS clone URL therefore contains two periods before `.git`.

## Contents

- [Requirements](#requirements)
- [Run the trained project locally](#run-the-trained-project-locally)
- [Run the API with Docker](#run-the-api-with-docker)
- [Train on Google Colab GPU](#train-on-google-colab-gpu)
- [Evaluate the models](#evaluate-the-models)
- [Run the API](#run-the-api)
- [Run the Streamlit demo](#run-the-streamlit-demo)
- [Architecture and model card](#architecture-and-model-card)
- [Project layout](#project-layout)
- [Notes and limitations](#notes-and-limitations)

## Requirements

- Python 3.10 or newer.
- Git.
- Docker Desktop with Docker Compose (for the containerized API).
- ICDAR 2003 dataset for preparing data and training. The dataset is not included in this repository;
  obtain it from an authorized source and follow its license/terms.
- A CUDA-capable NVIDIA GPU is recommended for training. Training on CPU is supported but can take
  substantially longer.

## Run the trained project locally

These instructions are for Windows PowerShell. The model checkpoints are not in GitHub; download
trained checkpoints separately (see [Train on Google Colab GPU](#train-on-google-colab-gpu)).

### 1. Clone and install

```powershell
git clone "https://github.com/anhvinh205/scene-text-recognition..git"
Set-Location ".\scene-text-recognition."
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"
```

If PowerShell blocks virtual-environment activation, either allow it for this terminal with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again, or run the venv
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

### 2. Put trained checkpoints in the expected locations

Copy the trained detector and recognizer files to:

```text
runs/detect/train/weights/best.pt
runs/recognition/best.pt
```

The paths are configured in `configs/pipeline.yaml`. If your detector checkpoint is elsewhere,
update `detector_weights` in that file to its actual path. Checkpoints, datasets, and generated
training outputs are intentionally excluded from Git.

### 3. Check the model and run a prediction

```powershell
Test-Path .\runs\detect\train\weights\best.pt
Test-Path .\runs\recognition\best.pt
```

Both commands should print `True`. Then test an image:

```powershell
scenetext predict .\path\to\your-image.jpg --output .\prediction.png
```

The recognized text and bounding boxes are printed to the terminal; the annotated image is saved as
`prediction.png`.

## Run the API with Docker

The Docker image runs inference on CPU. Docker packages the API and its dependencies; it does not
include the dataset or trained weights. The Compose file mounts the local `runs/` folder read-only,
so download the detector and recognizer checkpoints first and put them at the paths configured in
`configs/pipeline.yaml`:

```text
runs/detect/train/weights/best.pt
runs/recognition/best.pt
```

Install and start Docker Desktop, wait until it reports that the Docker Engine is running, then open
PowerShell in the project root. Set a private API key for this terminal and build/start the service:

```powershell
$env:SCENETEXT_API_KEY = "replace-this-with-a-private-key"
docker compose up --build -d
```

The first build downloads the base image and Python dependencies and may take several minutes. The
API loads both models when it starts; on CPU, wait for model loading to finish before checking its
health:

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

The health response should show `model_loaded: true` and `device: "cpu"`. Open
`http://localhost:8000/docs` to try the API. In Swagger, click **Authorize** and enter the same API
key set in `SCENETEXT_API_KEY`; do not commit or share this key.

To view startup or inference logs:

```powershell
docker compose logs -f api
```

Press Ctrl+C to stop following logs, then stop the container when finished:

```powershell
docker compose down
```

If Compose says `SCENETEXT_API_KEY` is not set, set it in the current PowerShell window again. If
the health response says `model_loaded: false` or startup reports a missing checkpoint, verify the
two checkpoint paths above. The Docker image and local CPU inference do not require a paid cloud
service.

## Train on Google Colab GPU

Training can be performed in a notebook on [Google Colab](https://colab.research.google.com). The
Google Colab VS Code extension is optional; using the Colab website directly is often simpler.

### 1. Create a GPU notebook

In Colab, create a notebook, then select **Runtime > Change runtime type > T4 GPU** (or another
available GPU) and save. Verify that the notebook actually received a GPU:

```python
!nvidia-smi

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Do not start training until `nvidia-smi` shows a GPU and PyTorch reports `CUDA available: True`.
GPU availability depends on the Colab account and current quota.

### 2. Clone the project and install dependencies

Run these commands in a notebook cell:

```python
%cd /content
!git clone "https://github.com/anhvinh205/scene-text-recognition..git" /content/scene-text-recognition
%cd /content/scene-text-recognition
!pip install -e .
```

### 3. Upload and prepare the dataset

Upload the authorized ICDAR 2003 ZIP file using Colab's **Files** panel. If the uploaded file is
named `icdar2003.zip`, run:

```python
%cd /content/scene-text-recognition
!mkdir -p data
!unzip -oq /content/icdar2003.zip -d data
!find data -name words.xml
```

The expected annotation path is
`/content/scene-text-recognition/data/SceneTrialTrain/words.xml`. If `find` prints a different
location, check the ZIP's folder structure and update `raw_dir` in `configs/data.yaml` before
preparing data.

Create the detector images and recognizer word crops:

```python
!scenetext prepare-data
```

The training data and generated crops live in the Colab runtime. They are temporary and can be lost
when the runtime disconnects or resets.

### 4. Train the detector and recognizer

Train the YOLO26 detector on GPU 0:

```python
!scenetext train-det --device 0
```

Train the CRNN recognizer on CUDA:

```python
!scenetext train-rec --device cuda
```

Training may take a while. Keep the runtime connected until each cell finishes. Colab sessions have
usage limits and may disconnect; save/download the checkpoints promptly after training.

The detector's output directory is controlled by `project` and `name` in
`configs/detection.yaml`. Find its actual checkpoint with:

```python
!find runs/detect -path "*/weights/best.pt"
!ls -lh runs/recognition/best.pt
```

The recognizer is expected at `runs/recognition/best.pt`.

## Evaluate the models

Evaluate detector precision/recall/mAP, passing the actual detector checkpoint path found above:

```python
!scenetext eval-det --weights runs/detect/runs/detect/train/weights/best.pt
```

If `find` returned another path, replace the example path above. Evaluate the recognizer:

```python
!scenetext eval-rec --device cuda
```

For end-to-end evaluation, create a Colab-specific pipeline config with the actual detector path:

```python
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text())
cfg["detector_weights"] = (
    "/content/scene-text-recognition/runs/detect/runs/detect/train/weights/best.pt"
)
cfg["recognizer_weights"] = "/content/scene-text-recognition/runs/recognition/best.pt"
cfg["device"] = "cuda"
Path("/content/pipeline_colab.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
```

Replace the detector path if your checkpoint is in a different directory, then run:

```python
!scenetext eval-e2e --config /content/pipeline_colab.yaml --yolo-dir data/yolo_data --split test --iou 0.5
```

## Track experiments with MLflow (optional)

MLflow runs locally and does not require an MLflow account or paid tracking service. Tracking is
disabled by default. Enable it separately in `configs/detection.yaml` and
`configs/recognition.yaml` by changing `tracking.enabled` to `true`. Both configs use
`sqlite:///mlflow.db`, so run metadata is stored in a local SQLite database and ignored by Git.
MLflow model and plot artifacts are stored locally under `mlartifacts/`, which is also ignored.

Install the optional dependency in the environment used for training:

```bash
pip install -e ".[tracking]"
```

For a Colab notebook, run the install after cloning the repository:

```python
%cd /content/scene-text-recognition
!pip install -e ".[tracking]"
```

Then edit both YAML files in the Colab file browser or in a notebook cell:

```python
from pathlib import Path
import yaml

for config_path in ("configs/detection.yaml", "configs/recognition.yaml"):
    path = Path(config_path)
    config = yaml.safe_load(path.read_text())
    config.setdefault("tracking", {})["enabled"] = True
    path.write_text(yaml.safe_dump(config, sort_keys=False))
```

Run training as usual. MLflow records detector parameters, per-epoch validation metrics and
Ultralytics training artifacts. For recognition it records parameters, train/validation metrics per
epoch, final test metrics, and the best checkpoint, history, and learning curves. Runs are grouped
into `scene-text-detection` and `scene-text-recognition` experiments.

To view runs locally, from the project root in a second terminal run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. If training took place in Colab, its `/content` storage is temporary. Before the runtime disconnects,
download both `mlflow.db` and the `mlartifacts/` directory and place them in the local project root;
then start the UI locally. Do not push these files to GitHub; model artifacts can be large, and
`.gitignore` excludes them.

To download them together from a Colab notebook:

```python
from pathlib import Path
import sqlite3
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

root = Path("/content/scene-text-recognition")
snapshot = Path("/content/mlflow-export.db")
with sqlite3.connect(root / "mlflow.db") as source, sqlite3.connect(snapshot) as target:
    source.backup(target)

archive = Path("/content/mlflow-results.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    z.write(snapshot, "mlflow.db")
    for path in (root / "mlartifacts").rglob("*"):
        if path.is_file():
            z.write(path, path.relative_to(root))
files.download(str(archive))
```

Extract `mlflow-results.zip` into the local project root. The archive may be large because it
contains model checkpoints logged as MLflow artifacts.

### Download the trained checkpoints

Package the two model files into one ZIP and download it from Colab to your computer:

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files

root = Path("/content/scene-text-recognition")
detector = root / "runs/detect/runs/detect/train/weights/best.pt"
recognizer = root / "runs/recognition/best.pt"
archive = Path("/content/scenetext-trained-models.zip")

assert detector.is_file(), f"Detector checkpoint not found: {detector}"
assert recognizer.is_file(), f"Recognizer checkpoint not found: {recognizer}"

with ZipFile(archive, "w") as z:
    z.write(detector, "runs/detect/train/weights/best.pt")
    z.write(recognizer, "runs/recognition/best.pt")

files.download(str(archive))
```

On Windows PowerShell, extract the downloaded ZIP into the project root:

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\scenetext-trained-models.zip" `
  -DestinationPath "C:\path\to\scene-text-recognition." -Force
```

Use the actual project path for `-DestinationPath`. The extracted files should be
`runs\detect\train\weights\best.pt` and `runs\recognition\best.pt` beneath that directory.
Do not commit model weights or the dataset to Git; they are large and are ignored intentionally.

## Run the API

Open a **new terminal** at the project root. In Windows PowerShell, activate the virtual
environment, set an API key in this terminal, and start the service:

```powershell
.\.venv\Scripts\Activate.ps1
$env:SCENETEXT_API_KEY = "replace-this-with-a-private-local-key"
scenetext serve
```

Keep this terminal open while using the API. The environment variable must be set in the same
terminal before starting the server. Do not publish a real API key in source code or screenshots.

Open `http://localhost:8000/health`. A successful response includes `"status":"ok"` and
`"model_loaded":true`. The `"device"` field reports where inference runs; for example, `"cpu"` means
the app is using the local machine's CPU, even if training was done on a Colab GPU.

To try OCR:

1. Open `http://localhost:8000/docs`.
2. Expand **POST /predict**, click **Try it out**, and choose a JPG or PNG image.
3. Click **Authorize** and enter the same key set in the server terminal. If prompted for the
   HTTP Bearer value, enter the key itself, not the word `Bearer`.
4. Click **Execute**. The JSON response includes recognized text, boxes, confidence scores, and
   latency.

If the response says `API authentication is not configured`, stop the server with Ctrl+C, set
`$env:SCENETEXT_API_KEY` in that same terminal, and restart `scenetext serve`. A `favicon.ico` 404
in the server log is harmless.

## Run the Streamlit demo

Leave the API terminal running if you still need it. Open a **second terminal** at the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`) and upload a JPG or PNG.
Streamlit reads both model paths from `configs/pipeline.yaml`.

## Architecture and model card

The detector used by the current project is **Ultralytics YOLO26** (`yolo26m.pt`), fine-tuned for
text regions. The recognizer is a CRNN with a ResNet34 backbone and BiLSTM. See
[the architecture notes](docs/architecture.md) for design details and
[the model card](docs/model_card.md) for dataset limitations and measured test results.

## Project layout

```text
configs/                Data, detector, recognizer, and inference configuration
src/scenetext/data/     ICDAR parser, dataset builder, crops, transforms, vocabulary, splits
src/scenetext/models/   CRNN model
src/scenetext/detection/ YOLO training and evaluation
src/scenetext/recognition/ CRNN training, evaluation, decoding, checkpoints
src/scenetext/pipeline.py End-to-end OCR pipeline
src/scenetext/serving/  FastAPI application
app/streamlit_app.py    Streamlit image-upload demo
tests/                  Automated tests
docs/                   Architecture, model card, and course answers
```

## Development and tests

```powershell
pip install -e ".[all]"
pytest
ruff check .
```

## Example training results

One run on the ICDAR 2003 split generated by `scenetext prepare-data` reported:

| Evaluation | Metric | Result |
|---|---|---:|
| Detector | Precision | 0.886 |
| Detector | Recall | 0.840 |
| Detector | mAP50 | 0.860 |
| Detector | mAP50-95 | 0.670 |
| Recognizer | Word accuracy | 0.523 |
| Recognizer | Character error rate (CER) | 0.127 |
| End-to-end pipeline | F1 at IoU 0.5 | 0.433 |

These are results from one training run, not a performance guarantee. Results can vary with the
runtime, dependency versions, dataset contents, and training configuration.

## Notes and limitations

- ICDAR 2003 SceneTrialTrain is a small dataset; results may not generalize to other cameras,
  languages, fonts, or real-world scenes.
- The default vocabulary is lowercase `0-9a-z`; uppercase letters and punctuation are not modeled.
- A Colab GPU is for training in the Colab session. A local API reports and uses the device available
  on the local machine.
- Keep large datasets, generated files, checkpoints, and secrets out of Git. Use an appropriate
  model-artifact service or Git LFS if you need to distribute model weights.
- Check the dataset license/terms before using or redistributing ICDAR 2003.

## License

MIT. See [LICENSE](LICENSE).
