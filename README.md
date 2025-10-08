# Project Oracle: NFL Big Data Bowl 2026

This repository contains the code for Project Oracle, a deep learning model developed for the NFL Big Data Bowl 2026. The project's core is a Transformer-based model named `OracleTransformer`, designed to predict the trajectories of NFL players during passing plays.

## Project Goal

The primary objective of this project is to predict the movement of all 22 players on the field for the 5-second interval immediately following the quarterback's pass release. The model takes pre-throw player tracking data as input and generates the predicted paths for all players.

## Repository Structure

The repository is organized as follows:

```
nflbigdatabowl/
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── supplementary_data.csv
│   └── processed/
├── models/
│   └── best_oracle_model_full.pth
├── notebooks/
│   └── eda.ipynb
├── reports/
│   └── figures/
│       └── full_training_loss_curve.png
├── results/
│   └── inference_results.parquet
├── src/
│   ├── data_processing.py
│   ├── dataset.py
│   ├── inference.py
│   ├── metrics.py
│   ├── model.py
│   ├── run_data_pipeline.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

  - **`data/`**: Contains raw and processed datasets.
  - **`models/`**: Stores the trained model weights.
  - **`notebooks/`**: Includes Jupyter notebooks for exploratory data analysis (EDA).
  - **`reports/`**: Contains generated reports and figures, such as training loss curves.
  - **`results/`**: Stores the output from the inference pipeline.
  - **`src/`**: Contains all the Python source code for the project.
  - **`requirements.txt`**: A list of all Python dependencies for the project.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  - Python 3.8+
  - An environment with `pip` for package management.
  - If you have a CUDA-enabled GPU, the training and inference processes will be significantly faster. The scripts are configured to automatically use a GPU if one is available.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gretareitenbach/nflbigdatabowl.git
    cd nflbigdatabowl
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project pipeline is broken down into three main stages: data processing, model training, and inference.

### 1\. Data Processing

This stage cleans the raw tracking data, standardizes coordinates so all plays move from left to right, merges supplementary information, and filters out invalid plays. The processed data is then saved in the more efficient `.parquet` format.

To run the full data pipeline, execute the following script from the root directory:

```bash
python src/run_data_pipeline.py
```

This will process all raw weekly CSV files located in `data/raw/train/` and save the output to `data/processed/`.

### 2\. Model Training

After processing the data, you can train the `OracleTransformer` model. The training script handles the creation of a training and validation set, initializes the model, and runs the training loop.

The best performing model based on validation loss will be saved to `models/best_oracle_model_full.pth`.

To start the training process, run:

```bash
python src/train.py
```

*Note: Training can be computationally intensive and may take a significant amount of time, especially without a GPU.*

### 3\. Inference

Once the model is trained, you can use it to generate predictions on the validation set. The inference script loads the best model and produces predicted player paths. It also saves the player order for each play to ensure correct analysis.

To run the inference pipeline, execute:

```bash
python src/inference.py
```

The results, including predicted paths and actual paths, will be saved to `results/inference_results.parquet`.

## Exploratory Data Analysis

The `notebooks/eda.ipynb` notebook provides a detailed exploratory analysis of the dataset. It includes data loading, initial inspections, descriptive statistics, and visualizations of player paths on a football field. This notebook is a great starting point for understanding the structure and nuances of the NFL tracking data.

## Model Architecture

The core of this project is the `OracleTransformer`, a Transformer-based neural network defined in `src/model.py`. It utilizes an Encoder-Decoder architecture to process the input sequence of player kinematics and generate the predicted output sequence. The model also incorporates a player embedding layer to create a unique vector representation for each player, allowing it to learn player-specific movement patterns.

## Key Scripts

  - **`src/run_data_pipeline.py`**: The main script to execute the entire data preparation process.
  - **`src/train.py`**: The primary script for training the model on the full dataset.
  - **`src/inference.py`**: Used to generate predictions with the trained model.
  - **`src/data_processing.py`**: Contains functions for cleaning and standardizing the data.
  - **`src/dataset.py`**: Defines the custom PyTorch `Dataset` for loading play data.
  - **`src/model.py`**: Contains the `OracleTransformer` model architecture.
  - **`src/metrics.py`**: Includes functions for evaluating model performance, such as Dynamic Time Warping (DTW).
