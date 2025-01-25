# Recommender System with Bayesian Personalized Ranking (BPR)

A machine learning model to recommend movies to users based on their past viewing history and preferences. This implementation leverages Bayesian Personalized Ranking (BPR) with Graph Neural Networks (GNNs).

---

## Project Structure

```plaintext
bpr_project/
├── data/
│   ├── download.py         # Functions for downloading and processing the dataset
│   ├── preprocess.py       # Functions for preprocessing (Spark/Pandas)
├── models/
│   ├── gnn.py              # GNN and Model classes
│   ├── loss.py             # BPR loss and evaluation metrics
├── utils/
│   ├── sampler.py          # Functions for BPR sampling
│   ├── metrics.py          # Helper functions for AUC and evaluation
│   ├── config.py           # Default configuration and hyperparameters
├── main.py                 # Main training script
├── train.py                # Training and validation loops
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

---

## File Descriptions

### Data

#### `data/download.py`
Functions for downloading and processing the dataset. These include methods for fetching raw data, validating its structure, and saving it in a structured format.

#### `data/preprocess.py`
Preprocessing utilities using Spark or Pandas. This script prepares the data for model training by cleaning, normalizing, and transforming the input data.

---

### Models

#### `models/gnn.py`
Implements the Graph Neural Network (GNN) and associated model classes. This module contains:
- Layers and architectures optimized for recommendation systems.
- Integrations with BPR for personalized ranking.

#### `models/loss.py`
Defines the Bayesian Personalized Ranking (BPR) loss function and evaluation metrics. This includes methods to calculate user-item pair rankings and optimize model performance.

---

### Utils

#### `utils/sampler.py`
Functions for Bayesian Personalized Ranking (BPR) sampling. This ensures efficient and effective sampling of user-item pairs for training.

#### `utils/metrics.py`
Helper functions for calculating metrics such as Area Under the Curve (AUC), hit rate, and precision for model evaluation.

#### `utils/config.py`
Contains default configuration settings and hyperparameters such as learning rate, batch size, and number of epochs.

---

### Core Scripts

#### `main.py`
The main entry point for training and evaluating the model. This script:
- Loads configurations.
- Initializes the model.
- Triggers training and evaluation workflows.

#### `train.py`
Contains the training and validation loops. Includes utilities for:
- Iterating through datasets.
- Calculating losses and updating model weights.
- Validating model performance after each epoch.

---

### Others

#### `requirements.txt`
Specifies the dependencies required for the project. Includes libraries like PyTorch, Pandas, NumPy, and PySpark.

#### `README.md`
Documentation providing an overview of the project, setup instructions, and usage examples.

---

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and preprocess data**:
   ```bash
   python data/download.py
   python data/preprocess.py
   ```

3. **Train the model**:
   ```bash
   python main.py
   ```

4. **Evaluate the model**:
   Use metrics provided in `utils/metrics.py` to assess the model's performance.

---

