# --- NFL Big Data Bowl 2026: Project Oracle ---
# Phase 2.3: Final Training & Validation Pipeline
#
# This is the main, production-ready script for training the Oracle model.
# It uses the entire dataset, a random train/validation split, a learning
# rate scheduler, and generates a final loss curve.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use relative imports
from .dataset import NFLTrackingDataset
from .model import OracleTransformer

def setup_dataloaders(batch_size: int, processed_data_dir: Path, raw_train_dir: Path, player_id_map: dict, train_val_split_ratio: float, random_seed: int):
    """
    Sets up the training and validation DataLoaders by creating a random split
    of the ENTIRE dataset.
    """
    # 1. Find all processed files
    all_processed_files = sorted(list(processed_data_dir.glob("processed_w*.parquet")))
    print(f"Found {len(all_processed_files)} processed weekly files.")

    # 2. Create a list of Dataset objects for all weeks
    print("\nInstantiating all weekly datasets...")
    all_datasets = [NFLTrackingDataset(
        processed_input_path=f,
        raw_output_dir=raw_train_dir,
        player_id_map=player_id_map
    ) for f in all_processed_files]

    # 3. Combine them into one single, massive dataset
    full_dataset = ConcatDataset(all_datasets)
    print(f"Combined all weeks into a single dataset with {len(full_dataset):,} plays.")

    # 4. Perform a random train/validation split
    print(f"Performing a {train_val_split_ratio:.0%} / {1-train_val_split_ratio:.0%} random split...")
    train_size = int(train_val_split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Use a generator with a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"\nTotal training plays: {len(train_dataset):,}")
    print(f"Total validation plays: {len(val_dataset):,}")

    # 5. Create the DataLoaders
    print("\nCreating DataLoaders...")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print("DataLoaders created successfully with a random split.")

    return train_loader, val_loader

def main():
    """ Main function to orchestrate the training process. """

    # --- 1. Configuration & Hyperparameters ---
    print("--- Setting up Training Configuration ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    MODEL_SAVE_PATH = PROJECT_ROOT / "models"
    MODEL_SAVE_PATH.mkdir(exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(exist_ok=True)

    # Data hyperparameters
    BATCH_SIZE = 32
    TRAIN_VAL_SPLIT_RATIO = 0.85 # Use 85% of the data for training
    RANDOM_SEED = 42 # For reproducible splits
    torch.manual_seed(RANDOM_SEED)

    # Model hyperparameters
    MODEL_DIM = 256
    N_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    EMBEDDING_DIM = 16

    # Training hyperparameters
    LEARNING_RATE = 1e-4
    EPOCHS = 40

    # --- 2. Create Player ID Map ---
    print("\nScanning for all unique players to create ID map...")
    all_input_files = PROCESSED_DATA_DIR.glob("*.parquet")
    all_player_ids = pd.concat([pd.read_parquet(f)['nfl_id'] for f in all_input_files]).unique()
    player_id_map = {nfl_id: i + 1 for i, nfl_id in enumerate(all_player_ids)}
    NUM_UNIQUE_PLAYERS = len(all_player_ids)
    print(f"Total unique players found across all weeks: {NUM_UNIQUE_PLAYERS}")

    # --- 3. Data Loading ---
    train_loader, val_loader = setup_dataloaders(
        BATCH_SIZE, PROCESSED_DATA_DIR, RAW_TRAIN_DIR, player_id_map,
        TRAIN_VAL_SPLIT_RATIO, RANDOM_SEED
    )

    # --- 4. Model, Loss, and Optimizer Setup ---
    kinematic_features = 6
    combined_player_features = kinematic_features + EMBEDDING_DIM
    model_input_dim = 22 * combined_player_features

    model = OracleTransformer(
        input_dim=model_input_dim,
        model_dim=MODEL_DIM,
        nhead=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_players_total=NUM_UNIQUE_PLAYERS + 1,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 5. Main Training Loop ---
    print("\n--- Starting Model Training (Full Run) ---")
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02} [Train]"):
            src = batch['X'].to(DEVICE)
            tgt = batch['y'].to(DEVICE)
            player_ids = batch['player_ids'].to(DEVICE)

            prediction = model(src, player_ids, tgt)
            loss = loss_fn(prediction, tgt)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_rmse = math.sqrt(avg_train_loss)
        train_loss_history.append(train_rmse)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:02} [Val]"):
                src = batch['X'].to(DEVICE)
                tgt = batch['y'].to(DEVICE)
                player_ids = batch['player_ids'].to(DEVICE)

                prediction = model(src, player_ids, tgt)
                loss = loss_fn(prediction, tgt)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_rmse = math.sqrt(avg_val_loss)
        val_loss_history.append(val_rmse)

        scheduler.step(val_rmse)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f'\tTrain RMSE: {train_rmse:.3f}')
        print(f'\t Val. RMSE: {val_rmse:.3f}')

        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            torch.save(model.state_dict(), MODEL_SAVE_PATH / 'best_oracle_model_full.pth')
            print(f"\t-> New best model saved with validation RMSE: {best_val_loss:.3f}")

    print("\n--- Model Training (Full Run) Complete ---")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, EPOCHS + 1), train_loss_history, 'b-o', label='Training RMSE')
    plt.plot(range(1, EPOCHS + 1), val_loss_history, 'r-o', label='Validation RMSE')
    plt.title('Full Training & Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (yards)')
    plt.legend()
    plt.grid(True)
    loss_curve_path = REPORTS_DIR / "figures" / "full_training_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=150)
    print(f"\nLoss curve plot saved to: {loss_curve_path}")

if __name__ == "__main__":
    main()
