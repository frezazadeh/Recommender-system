[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)


# Movie Recommendation System

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Downlaod and Preprocess Data and Run Training](#downlaod-and-preprocess-data-and-run-training)
5. [Training and Validation Metrics Analysis for Big Data](#training-and-Validation-Metrics-Analysis-for-Big-Data)
---

# Problem Statement

The goal of this project is to develop a machine learning model that recommends movies to users based on their past viewing history and preferences. Using the MovieLens dataset, the system aims to provide personalized movie recommendations through a machine learning model integrated into a Fast API.

Graph Neural Networks (GNNs) are used to process the graph-structured data where nodes represent entities (e.g., users and movies), and edges represent interactions (e.g., ratings). In my code, GNNs are the backbone of the recommendation model.

Bayesian Personalized Ranking (BPR) optimizes the recommendation model by learning to rank observed (positive) user-movie interactions higher than unobserved (negative) ones.

---

# Project Structure

```plaintext
gnn-bpr_project/
├── saved_model/
├── plots/
├── data/
│   ├── download.py         # Functions for downloading and processing the dataset (Spark/Pandas)
├── models/
│   ├── gnn.py              # GNN and Model classes
│   ├── loss.py             # BPR loss and evaluation metrics
│   ├── tarin.py            # Training and validation loops (core of application)
├── utils/
│   ├── sampler.py          # Functions for BPR sampling
│   ├── metrics.py          # Helper functions for AUC and evaluation
├── main.py                 # Main training script
├── app.py                  # API Gateway for System (FastAPI)
├── frontend.py             # User interface for interacting with the backend movie recommendation system
├── requirements.txt        # Dependencies
```

---
# Installation

Clone this repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Downlaod and Preprocess Data and Run Training

You can run the main script with the following options:

```bash
python main.py [OPTIONS]
```

### Arguments

| Argument           | Type    | Default   | Description                                          |
|--------------------|---------|-----------|------------------------------------------------------|
| `--data_size`      | `str`   | `small`   | The size of the dataset to use (`big` or `small`).   |
| `--hidden_channels`| `int`   | `32`      | Number of hidden channels in the model.             |
| `--lr`             | `float` | `1e-3`    | Learning rate for the optimizer.                    |
| `--batch_size`     | `int`   | `1024`    | Batch size for training.                            |
| `--epochs`         | `int`   | `150`     | Number of training epochs.                          |
| `--steps`          | `int`   | `100`     | Number of steps for intermediate logging or updates.|
| `--weight_decay`   | `float` | `1e-5`    | Weight decay for regularization.                    |
| `--patience`       | `float` | `30`      | Early stopping patience.                            |

### Examples

Run the code with the default settings:

```bash
python main.py
```

Run the code with a big dataset and a custom learning rate:

```bash
python main.py --data_size big --lr 0.001
```

Change the batch size and number of epochs:

```bash
python main.py --batch_size 512 --epochs 200
```
The backend is implemented using FastAPI. To run the backend server, use the following command:

```bash
uvicorn app:app --reload
```

- `app:app`: This specifies the file (`app.py`) and the FastAPI application instance (`app`) to run.
- `--reload`: Enables automatic reloading of the server whenever code changes are detected, which is useful during development.

The frontend is implemented using Streamlit. To start the frontend application, use the following command:

```bash
streamlit run frontend.py
```

- `frontend.py`: The Streamlit script containing the frontend logic.
- After running this command, a local server will start, and a URL will be displayed in the terminal (e.g., `http://localhost:8501`). Open this URL in your web browser to interact with the frontend.

Below is a screenshot of the frontend interface:

<img src="plots/Frontend.png"/>

--- 

# Training and Validation Metrics Analysis for Big Data

## Overview
The plots below illustrate the progression of key metrics during the training of a recommendation model using Bayesian Personalized Ranking (BPR) loss. These metrics include:

1. **BPR Loss**: Tracks the loss values for both the training and test datasets over epochs.
2. **Validation AUC**: Measures the model's ability to rank items correctly on a validation dataset.

## Metrics Analysis

### 1. **Train vs. Test BPR Loss**
- **Description**: The left plot shows the training and test BPR loss across epochs.
- **Insights**:
  - Both training and test loss decrease rapidly in the early epochs, indicating effective learning.
  - After approximately 40 epochs, the loss stabilizes near 0.05 for both training and test datasets.
  - Minimal divergence between training and test loss demonstrates good generalization and minimal overfitting.

### 2. **Validation AUC**
- **Description**: The right plot tracks the AUC (Area Under the Curve) on the validation dataset over epochs.
- **Insights**:
  - AUC improves significantly in the initial epochs, surpassing 0.975 after around 30 epochs.
  - The metric continues to improve gradually, stabilizing near 0.985–0.990 after 100 epochs.
  - Consistent improvement without noticeable drops suggests robust model performance on unseen data.

## Conclusions
- The model converges effectively, with both BPR loss and AUC stabilizing after approximately 100 epochs.
- The alignment between training and test BPR loss highlights a lack of overfitting, indicating that the model can generalize well.
- High AUC values (close to 0.98) suggest excellent ranking performance on the validation set.

**Generated Metrics Plot:**

<img src="plots/training_validation_metrics_big.png"/>

---

# Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request.

---

# Acknowledgement

[Services as NetworkS (SaS)](https://www.cttc.cat/services-as-networks-sas/)


