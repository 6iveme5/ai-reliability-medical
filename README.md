# Density-Based Reliability Estimation for Medical AI

This repository contains code for evaluating pointwise reliability estimation in medical AI using Gaussian Mixture Models (GMMs) in centralized and federated learning settings.

The project trains an MLP classifier on tabular medical features, extracts latent embeddings, and evaluates reliability scores for misclassification detection.

## Project Structure

```text
centralized/        Centralized training and reliability evaluation
federated/          Federated learning with FedAvg and reliability evaluation
configs/            Shared experiment configuration
models/             MLP classifier, GMM reliability, and TrustScore
utils/              Data loading, splitting, scaling, and DataLoader utilities
data/               Local dataset directory, not tracked by Git
results/            Generated outputs, not tracked by Git
```

## Main Methods

- Softmax confidence baseline
- TrustScore in embedding space for centralized comparison
- Class-wise GMM reliability in learned embedding space
- FedAvg-based federated classifier training

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ai_reliability_medical
```

## Data

Raw and processed medical data are not included in this repository. Place local CSV files in the following structure:

```text
data/
  centralized/
    trainData_cleaned.csv
  test/
    testData_cleaned.csv
  federated/
    client_1.csv
    client_2.csv
    ...
```

The expected target column is `outcomeType`. Feature columns are defined in `configs/config.py`.

## Run Experiments

Centralized experiment:

```bash
python centralized/run_centralized.py
```

Federated experiment:

```bash
python federated/run_federated.py
```

Outputs are written under `results/`.

## Notes

- Do not commit private medical data, generated model checkpoints, or result artifacts unless they have been reviewed for sharing.
- The current federated GMM implementation should be reviewed before final publication because GMM fitting must avoid test-label leakage.
