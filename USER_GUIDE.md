# User Guide

This backend should be paired with the frontend at `https://github.com/anclark686/reyaly-disease-predictor`

**Note**: Large data files are intentionally ignored by Git.

## Overview

This project is a local disease prediction web application with:

- a `FastAPI` backend in `/backend`
- a `Bun + React` frontend in `/frontend`

When running locally:

- the backend runs on `http://localhost:8000`
- the frontend runs on `http://localhost:3000`

## Summary

To run the full project locally:

1. `cd backend`
2. `pipenv install`
3. `pipenv run uvicorn app.main:app --reload`
4. Open a second terminal
5. `cd frontend`
6. `bun install`
7. `bun dev`
8. Open `http://localhost:3000`


## First-Time Setup Checklist

Use the Windows section below for evaluation. The macOS section is included for
completeness.

## Prerequisites

Before starting, install the following tools:

- Python `3.14`
- `pipenv`
- `Bun`
- a Kaggle account to download the dataset

Helpful links:

- Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Pipenv: [https://pipenv.pypa.io/en/latest/installation.html](https://pipenv.pypa.io/en/latest/installation.html)
- Bun: [https://bun.com/docs/installation](https://bun.com/docs/installation)
- Dataset: [https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data)
- Heroku CLI: [https://devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

## Local Setup

<details open>
<summary><strong>Windows Setup (Recommended)</strong></summary>

Install the required tools:

1. Download Python `3.14` from the official Python website.
2. Run the installer and make sure `Add Python to PATH` is checked.
3. Verify Python:

```powershell
python --version
```

4. Install `pipenv`:

```powershell
python -m pip install --user pipenv
pipenv --version
```

5. Install Bun:

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
bun --version
```

Download the dataset from Kaggle:

[https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data)

After downloading it:

1. extract the archive on your machine
2. rename the file `Diseases and Symptoms Dataset.csv` to `diseases_symptoms.csv`
3. create a directory `backend/data/raw/` if it doesn't exist
4. place the raw CSV at `backend/data/raw/diseases_symptoms.csv`

Regenerate the processed data files:

```powershell
cd backend
pipenv run python scripts/preprocess_data.py
```

Open a terminal in `backend` and run:

```powershell
pipenv install
```

Open a second terminal in `frontend` and run:

```powershell
bun install
```

Start the backend:

```powershell
cd backend
pipenv run uvicorn app.main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
bun dev
```

Open:

```text
http://localhost:3000
```

### Optional

If you need to regenerate processed data files or retrain the model, use the
scripts in `backend/scripts/`.

Generated processed data files should be stored in:

```text
backend/data/processed/
```

If you update the source data or retrain the backend model, regenerate the
visualization artifacts before running or deploying the API:

```powershell
cd backend
pipenv run python scripts/generate_visualization_artifacts.py
```

</details>

<details>
<summary><strong>macOS Setup</strong></summary>

Install the required tools:

1. Download Python `3.14` from the official Python website.
2. Run the installer.
3. Verify Python:

```bash
python3 --version
```

4. Install `pipenv`:

```bash
python3 -m pip install --user pipenv
pipenv --version
```

5. Install Bun:

```bash
curl -fsSL https://bun.sh/install | bash
bun --version
```

Download the dataset from Kaggle:

[https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset/data)

After downloading it:

1. extract the archive on your machine
2. rename the file `Diseases and Symptoms Dataset.csv` to `diseases_symptoms.csv`
3. create a directory `backend/data/raw/` if it doesn't exist
4. place the raw CSV at `backend/data/raw/diseases_symptoms.csv`

Regenerate the processed data files:

```bash
cd backend
pipenv run python scripts/preprocess_data.py
```

Open a terminal in `backend` and run:

```bash
pipenv install
```

Open a second terminal in `frontend` and run:

```bash
bun install
```

Start the backend:

```bash
cd backend
pipenv run uvicorn app.main:app --reload
```

Start the frontend in a second terminal:

```bash
cd frontend
bun dev
```

Open:

```text
http://localhost:3000
```

### Optional

If you need to regenerate processed data files or retrain the model, use the
scripts in `backend/scripts/`.

Generated processed data files should be stored in:

```text
backend/data/processed/
```

If you update the source data or retrain the backend model, regenerate the
visualization artifacts before running or deploying the API:

```bash
cd backend
pipenv run python scripts/generate_visualization_artifacts.py
```

</details>

## Project Structure

- `/backend`
  Contains the FastAPI API, ML service code, and data-processing scripts.
- `/frontend`
  Contains the React user interface.
- `/backend/app/ml`
  Contains saved model artifacts used by the application.

## Running the App Locally

You will need two terminal windows:

1. one for the backend
2. one for the frontend

When the backend starts successfully, it should be available at:

```text
http://localhost:8000
```

You can verify the API is running by opening:

```text
http://localhost:8000/docs
```

When the frontend starts successfully, open:

```text
http://localhost:3000
```

## Stopping the Application

To stop either server:

- press `Ctrl + C` in the terminal window where it is running


## Recommended Startup Order

Start the project in this order:

1. Start the backend first.
2. Start the frontend second.
3. Open the frontend in your browser.

Starting the backend first helps prevent frontend API connection errors.

## Optional: Reinstall Dependencies

If you run into missing-package errors, reinstall dependencies.

### Backend

```bash
cd backend
pipenv install
```

### Frontend

```bash
cd frontend
bun install
```

## Optional: Frontend API Override

If you want the frontend to talk to a different backend URL, create a file named
`frontend/.env.local` and set:

```bash
BUN_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Common Issues

### Backend will not start

Possible causes:

- Python `3.14` is not installed
- `pipenv` is not installed
- backend dependencies were not installed yet

Try:

```bash
cd backend
pipenv install
pipenv run uvicorn app.main:app --reload
```

### Frontend will not start

Possible causes:

- Bun is not installed
- frontend dependencies were not installed yet

Try:

```bash
cd frontend
bun install
bun dev
```

### Frontend loads, but predictions or charts fail

Possible causes:

- backend server is not running
- backend is running on the wrong port

Check that the backend is available at:

```text
http://localhost:8000
```

If you are using a custom backend URL for the frontend, verify that
`BUN_PUBLIC_API_BASE_URL` points to the correct deployed backend.
