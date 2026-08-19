Voice Biometric Authentication

This repository contains a PyTorch-based speaker embedding / voice biometric backend. It includes training and evaluation scripts under `Voice_authentication/` and helper scripts.

Quick start (recommended):

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
# On Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `torch` installation depends on your CUDA/CPU platform. If you need GPU support, follow the official PyTorch install instructions for your environment and install the appropriate `torch`/`torchaudio` wheel.

2. Prepare data:

- Download the LibriSpeech test/dev data and place it under `audio-dev-clean/LibriSpeech/` and `LibriSpeech/test-clean/` to match paths in CSVs.
- Place a trained model file (for example `CNN.pt`) under `models/` or pass `--model` to the evaluation script.

3. Run evaluation (example):

```bash
python Voice_authentication/evaluate.py --model models/CNN.pt --train Train_data.csv --test Test_data.csv
```

If your CSVs reference Google Colab mount paths, either run the evaluation script with `--normalize-csvs` to update CSVs in-place, or run `python scripts/normalize_csvs.py` to create normalized copies.

What I added to help forkers:
- `requirements.txt` with common dependencies
- `README.md` with setup/run instructions
- `scripts/check_imports.py` to verify required Python packages
- `scripts/normalize_csvs.py` to convert Colab/GDrive paths in CSVs to repo-local paths
- `Voice_authentication/evaluate.py` now accepts CLI args and standardizes CSV columns before evaluation

Notes about pushing to GitHub:
- Large model files (`*.pt`) and the full LibriSpeech dataset are large — consider using `git-lfs` or hosting models elsewhere. This repo intentionally keeps paths flexible so forkers can point to local models/datasets.

If you want, I can:
- Add a small FastAPI wrapper for inference
- Add a CI workflow (GitHub Actions) that runs import checks
- Add `gitignore` and `CONTRIBUTING.md` before you push

Tell me which of the above you'd like next.

**Git & Large Files**

- **Use Git LFS for model files**: model checkpoints (`*.pt`) are large — use Git Large File Storage to track them instead of committing raw binaries into the repo history. Quick setup:

```bash
# install git-lfs (one-time)
git lfs install
# tell git-lfs to track PyTorch model files
git lfs track "*.pt"
# commit the generated .gitattributes
git add .gitattributes
git commit -m "Track .pt files with Git LFS"
```

- If you've already committed large model files and want to migrate them to LFS, follow the Git LFS documentation and consider removing them from history (use with caution):

```bash
git rm --cached models/*.pt
git commit -m "Remove large models from history; track with LFS instead"
git push
```

- I added a `.gitattributes` file to this repo to recommend LFS tracking for `*.pt` files. Do not add `*.pt` to `.gitignore` if you intend to track them with LFS; `.gitignore` should be used to avoid committing datasets, temporary files, and environment folders.

**Cleaning Colab / Google Drive artifacts**

Many notebooks and scripts originally written in Colab include drive-mounts or absolute mount paths. To help remove these safely, run the scanner that reports occurrences and can optionally comment them out (backups are created with `.bak`):

```bash
python scripts/find_colab_artifacts.py        # report-only
python scripts/find_colab_artifacts.py --fix # comment-out offending lines, create .bak backups
```

The repository also contains `scripts/normalize_csvs.py` which helps convert CSV references from Colab paths to local `LibriSpeech/` paths.

If you'd like, I can automatically strip Colab cells from specific notebooks or help rewrite them to use local paths instead of mount points.