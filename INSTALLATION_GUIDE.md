# Installation Guide

## Requirements

- Windows
- Python 3.10 or later
- pip
- Access to the project directory

## Open the project

Open PowerShell and run:

```powershell
cd "C:\Users\hp\Analysis in Global Supply Chain Operations"
```

## Create a virtual environment

```powershell
python -m venv .venv
```

## Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate execution policy permitted
by your computer administrator, or run the environment Python directly.

## Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Launch the dashboard

```powershell
streamlit run app.py
```

## Run notebooks

Start Jupyter from the project root so project-relative paths resolve correctly:

```powershell
jupyter notebook
```

## Troubleshooting

- Confirm that the project root contains `app.py`.
- Confirm that the raw workbook is in `data/raw`.
- Confirm that the processed CSV files are in `data/processed`.
- Check that the active Python environment has the packages in requirements.txt.
- Do not fabricate an operational date column when the source data does not
  contain one.
