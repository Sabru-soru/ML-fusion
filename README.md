# ML-fusion

This repository contains a collection of standalone Python scripts and data files for training and evaluating machine-learning models for fusion research.

## Repository structure
- `aktualno/` – scripts for the current workflow (data extraction, exploration and modeling)
- `potential_analysis/` – experimental notebooks and alternative approaches
- `OLD_code/` – legacy experiments and older scripts
- `data/` – pickled datasets and example spreadsheets
- `fig/` – generated figures and interactive HTML plots

## Getting started
1. Run `aktualno/0_extracting_data.py` to create `data/df_data.pkl` from the provided spreadsheets.
2. Use `aktualno/1_data_exploration.py` to inspect the dataset and generate interactive plots.
3. Train and evaluate models using the scripts under `aktualno/2_*` and `aktualno/3_*`.

The codebase is organized as individual scripts that can be run directly. See the `fig/` folder for examples of the output visualizations.
