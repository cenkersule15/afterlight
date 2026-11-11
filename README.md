# AFTERLIGHT

**Satellite imagery. Measured evidence.**

A local, trilingual geospatial ML research workspace by Cenker Sule. Compare real before/after imagery, inspect reference labels and run trained PyTorch / scikit-learn models. Includes a 66-pair xBD pilot subset, trained checkpoints and measured results. No API key or account is needed.

[Deutsch](README.de.md) · [Türkçe](README.tr.md)

## Studio 02 interface update

Dark and light themes, searchable scene library, pinned scenes stored in the browser, side-by-side comparison, focus mode, region class/area filters, selected-region inspection and scene-specific IoU/Dice/F1. Keyboard: Cmd/Ctrl+K opens the library; left/right changes the scene; R runs inference, L toggles reference labels, F toggles focus. These shortcuts are ignored in form fields. Model checkpoints, data partitions and reported aggregate scores are unchanged.


## Start in VS Code with Docker

1. Extract the ZIP and open the **afterlight** folder (the folder containing `compose.yaml`) in VS Code.
2. Open Docker Desktop and wait until the engine is running.
3. Open a terminal in that folder:

```bash
docker compose up --build
```

Open **http://localhost:8080**. API documentation: **http://localhost:8080/api/docs**.
The first build downloads Python, Node and CPU PyTorch dependencies and can take several minutes. No Node/Python installation is needed on the host when using Docker. CPU images support Linux AMD64 and ARM64; Docker Desktop runs the containers on Mac.

Stop: `Ctrl+C`, then `docker compose down`. Source images and checkpoints remain in the local `data/` and `artifacts/` bind mounts. Training is **not** performed at application startup.

## What works

- EN / TR / DE interface with saved language preference.
- Before/after slider, zoom, event/scene selection and reference-label overlays.
- Pair U-Net and Random Forest inference with damage overlays, connected-region inspection and JSON export.
- Upload aligned PNG/JPEG pairs with matching dimensions (8 MiB per file, ≤16 MP). Uploads are processed in memory; no reference metrics are invented for unlabelled uploads.
- Data laboratory: event-disjoint partitions and measured label-pixel class distributions.
- Model evaluation: real held-out metrics, confusion matrices, per-class F1/IoU and validation curve.

## Research result: small pilot, low generalisation

| Model | Damage macro F1 | Building IoU | Building Dice |
|---|---:|---:|---:|
| Pair U-Net | 0.1031 | 0.2401 | 0.3872 |
| Random Forest | 0.1105 | 0.1644 | 0.2824 |

These are low scores on a deliberately small, event-disjoint pilot. They are **not production-ready performance**. Random Forest slightly exceeds the CNN on damage macro F1; the CNN has higher building IoU. This is a reproducible starting experiment, not a claim that deep learning always wins. No emergency or building-safety decisions should use these results.

The six-channel Pair U-Net jointly segments background and four damage classes from paired RGB images. It is trained from scratch; this package does not claim a pretrained foundation model or two independently trained stages. Connected components are predicted **regions**, not reliable counts of individual buildings. Model scores are not calibrated probabilities.

## Data and evaluation

- Source: xBD / xView2, Gupta et al. / DIU / CMU SEI; imagery: Maxar Open Data Program.
- Mirror: https://huggingface.co/datasets/kshitijrajsharma/xview2-xbd
- Original project: https://xview2.org/ ; paper: https://arxiv.org/abs/1911.09296
- Pinned mirror revision, source checksum and transformations: `data/catalog.json`.
- 66 pairs from 9 events, selected from the mirror's first training shard, at most 8 pairs per event. This is a convenience subset, not a representative survey.
- Custom event-disjoint partitions: 37 train / 13 validation / 16 test. The whole named event stays in one partition. These are **not** the official xView2 challenge splits.
- Original 1024 px imagery is resized to 512 px in the bundle; models operate at 128 px. Masks use nearest-neighbour resampling. Unknown labels are ignored (255).
- Metrics pool the confusion matrix across held-out pixels. Foreground macro F1 excludes classes absent from both truth and prediction, which are listed in the report. These are not official challenge metrics or object-level accuracy.
- Validation selects the CNN checkpoint; test is evaluated after selection. Seed, hashes, history and event lists are stored in `artifacts/report.json`.

## Train separately

```bash
docker compose --profile training run --rm train
```

This trains both models, replacing the files in `artifacts/`. The bundled run uses seed 42, 40 epochs and 128 px inputs. CPU reproducibility can vary slightly by hardware/library kernels.
After training, reload the model files by restarting the API:

```bash
docker compose restart api
```

A longer experiment:

```bash
docker compose --profile training run --rm train python -m afterlight.train --epochs 80 --size 256
```

The UI reads the report's actual resolution. Higher resolution consumes more CPU and memory. This Docker setup is CPU-only; NVIDIA CUDA training is not configured, and Mac containers do not use Apple MPS. Training is a CLI process, not a background job system.

## Import a larger local xBD collection

Obtain the official data independently and respect its terms. Put extracted PNG/JSON pairs under `data/raw/xbd`:

```bash
docker compose --profile training run --rm -v "${PWD}/data:/workspace/data" train python -m afterlight.prepare --xbd /raw/xbd
```

Then retrain. This importer replaces the catalogue and retains event-based rules from `backend/afterlight/prepare.py`; review event coverage before training. The dataset hash detects a mismatch with old checkpoints. The raw files are not redistributed by this importer.

## Source-only GitHub clone

The supplied ZIP includes prepared data and weights. `.gitignore` excludes imagery, raw data and model binaries to keep a future Git repository small. After cloning source only, create a Python 3.12 environment, install `backend/requirements.txt` and CPU PyTorch 2.8.0, then run `python scripts/fetch_pilot.py` (about 509 MB source download) and retrain. The fetcher verifies the pinned source checksum. It is unnecessary for the complete ZIP.

## Verification

```bash
docker compose --profile training run --rm -v "${PWD}/backend/tests:/app/tests:ro" train python -m unittest discover -s /app/tests -v
```

Native authoring validation: 14 Python tests passed, TypeScript and Vite build passed, model training and inference executed on CPU. Docker is not available in the authoring environment: image builds and `docker compose up` require final verification on a Docker host. See `docs/VALIDATION.md`.

## Architecture

`data/catalog.json + PNG pairs → preprocessing → PyTorch / Random Forest → saved artifacts → FastAPI → React / Nginx`

No database is needed for this local read-only research workspace. User uploads are not saved. There is no live satellite feed, automatic image registration, geographic map or production authentication. Image rendering is in pixel coordinates.

## Licences

Original application code: MIT (`LICENSE`). Bundled xBD-derived images/masks **and bundled pilot model artifacts**: CC BY-NC-SA 4.0 (`data/LICENSE.md`, `artifacts/LICENSE.md`). Non-commercial use, attribution and share-alike terms apply to those assets; the code licence does not relicense the data. Retain attribution when sharing. UI fonts: Google Fonts DM Sans / Manrope, with system-font fallback if offline.

## Next research steps

Larger event-balanced data, higher-resolution patches, stronger pretrained encoders, calibration, independent geographic evaluation and object-level evaluation. Improve the evidence before presenting this as a high-performing CV model.
