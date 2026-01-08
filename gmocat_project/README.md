# GMOCAT Framework

Implementation of the GMOCAT (Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing) framework, supporting ASSISTments 2009 and DBE-KT22 datasets.

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The code expects datasets to be placed in the `data/` directory.

### 1. ASSISTments 2009
- Download the dataset (e.g., `skill_builder_data.csv` or `assistments_2009_2010.csv`).
- Rename it to `assistments_2009.csv`.
- Place it in: `data/assistments/assistments_2009.csv`.

### 2. DBE-KT22
- Obtain access to the dataset from the Australian Data Archive (ADA).
- Place the following CSV files in `data/dbekt22/raw/`:
    - `Transaction.csv`
    - `Question_KC_Relationships.csv`

## Running Experiments

To run the full suite of experiments (2 datasets x 3 seeds):

```bash
python3 run_experiments.py
```

Results will be saved to `results/experiment_summary.csv` and detailed logs in `logs/`.

## Project Structure

```
gmocat_project/
├── data/               # Dataset storage
├── src/                # Source code
│   ├── models/         # Neural models (GNN, NCDM, RL)
│   ├── config.py       # Configuration
│   ├── data_loader.py  # Data processing
│   ├── train.py        # Training loop
│   └── utils.py        # Metrics & Utilities
├── run_experiments.py  # Experiment runner script
└── results/            # Output metrics
```
