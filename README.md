# Agri-AI

A plant leaf disease detection project. The repository excludes large datasets, model checkpoints, and logs to keep the repo lightweight.

## What’s included
- Application code and metadata
- `.gitignore` configured to exclude large artifacts (datasets, models, logs, `node_modules`)

## What’s excluded (place locally)
- Datasets: place under `agriai/Plant_leave_diseases_dataset_without_augmentation/`
- Trained models/checkpoints: place under `agri_ai_final/models/` (e.g., `final_model.keras`)
- TensorBoard logs/checkpoints: place under their original folders (they are ignored)

These paths are ignored by Git and won’t be uploaded.

## Setup
1. Python environment
   - Python 3.10+
   - Install requirements in `agri_ai_final/requirements.txt` or `agriai/requirements.txt` as needed:
     - `pip install -r agri_ai_final/requirements.txt`
     - `pip install -r agriai/requirements.txt`

2. Frontend (optional, archived sample in `agri_ai_final/archived/agrinew-main/`)
   - Requires Node.js 18+
   - From `agri_ai_final/archived/agrinew-main/`: `npm install && npm run dev`

## Running the backend
- Typical entry points (depending on your setup):
  - `python agri_ai_final/api.py`
  - or `python agri_ai_final/run_app.py`
  - or `python agri_ai_final/start_backend.py`

If your entry point differs, update this README and your start script accordingly.

## Models and datasets
- Place your trained Keras model at `agri_ai_final/models/final_model.keras` (ignored)
- Place the dataset at `agriai/Plant_leave_diseases_dataset_without_augmentation/` (ignored)

If you want to distribute models/datasets:
- Upload to cloud storage or a GitHub Release and add the download link here.

## Notes
- Large artifacts are intentionally excluded to avoid multi-GB pushes and Git LFS requirements.
- If you need Git LFS, enable it and remove ignore rules for specific file types.
