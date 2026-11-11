# Validation — 2026-11-11

Executed on Linux / Python 3.12 / Node 24:

- Real xBD mirror source shard downloaded, pinned revision and source SHA-256 recorded.
- 66 PNG pairs with masks prepared across 9 named events.
- CPU Pair U-Net training (40 epochs, 128 px) and Random Forest fitting executed; validation checkpoint selection and held-out metrics saved.
- 14 Python tests passed: pair integrity, label IDs, event-disjoint groups, report/data hash, perfect-metric formula, API health/stats, unknown scene/model rejection, PNG overlay, both models' real inference, malformed uploads, dimension mismatch and valid upload without invented accuracy.
- TypeScript and Vite production build passed.
- Browser QA: catalogue, imagery loading, CNN inference and region output, reference/model layer distinction, model metrics and language changes.

Docker daemon/CLI is unavailable in this authoring environment. Dockerfiles and Compose are supplied; a real Docker image build and end-to-end Compose startup have NOT been executed here. Validate on Docker Desktop before treating this as a tested container deployment.

The GitHub workflow checks the frontend build, Python compilation and Compose syntax only. Full data-dependent tests require the included ZIP data/models or the documented data restoration and training. No GitHub Actions run is claimed.
