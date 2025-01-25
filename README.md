# Recommender-system
Develop a machine learning model to recommend movies to users based on their past viewing history and preferences.


## File Descriptions

### Data
- **`data/download.py`**: Functions for downloading and processing the dataset.
- **`data/preprocess.py`**: Functions for preprocessing the data using Spark or Pandas.

### Models
- **`models/gnn.py`**: Implements the Graph Neural Network (GNN) and model classes.
- **`models/loss.py`**: Defines the BPR loss function and evaluation metrics.

### Utils
- **`utils/sampler.py`**: Functions for Bayesian Personalized Ranking (BPR) sampling.
- **`utils/metrics.py`**: Helper functions for metrics such as AUC.
- **`utils/config.py`**: Default configuration and hyperparameters.

### Core Scripts
- **`main.py`**: Main entry point for training and evaluation.
- **`train.py`**: Contains the training and validation loops.

### Others
- **`requirements.txt`**: Specifies the dependencies required for the project.
- **`README.md`**: Documentation for the project.

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

