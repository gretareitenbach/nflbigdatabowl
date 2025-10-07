# --- NFL Big Data Bowl 2026: Project Oracle ---
# Phase 3.1: Inference Pipeline (Robust Version)
#
# This script loads the best trained model and uses it to generate "Optimal Path"
# predictions for the validation dataset. Crucially, it also saves the exact
# player order for each play to prevent ambiguity during analysis.

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from dataset import NFLTrackingDataset
from model import OracleTransformer

def main():
    """ Main function to run the inference process. """

    # --- 1. Configuration ---
    print("--- Setting up Inference Configuration ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
    MODEL_PATH = PROJECT_ROOT / "models" / "best_oracle_model_full.pth"
    RESULTS_SAVE_DIR = PROJECT_ROOT / "results"
    RESULTS_SAVE_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 64
    TRAIN_VAL_SPLIT_RATIO = 0.85
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    # --- 2. Load Data ---
    print("\nLoading all necessary data sources...")

    # We need the full tracking data to look up player order for each play
    all_processed_files = sorted(list(PROCESSED_DATA_DIR.glob("*.parquet")))
    full_tracking_df = pd.concat([pd.read_parquet(f) for f in all_processed_files])
    print(f"  - Loaded full tracking data with {len(full_tracking_df):,} rows.")

    # Create the player ID map (must be identical to the one used for training)
    all_player_ids = full_tracking_df['nfl_id'].unique()
    player_id_map = {nfl_id: i + 1 for i, nfl_id in enumerate(all_player_ids)}
    NUM_UNIQUE_PLAYERS = len(all_player_ids)

    # Instantiate all datasets and perform the same random split as in training to get the validation set
    all_datasets = [NFLTrackingDataset(f, RAW_TRAIN_DIR, player_id_map) for f in all_processed_files]
    full_dataset = ConcatDataset(all_datasets)
    train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  - Validation set loaded with {len(val_dataset):,} plays.")

    # --- 3. Load the Trained Model ---
    print(f"\nLoading trained model from: {MODEL_PATH}")
    # Re-instantiate the model with the same hyperparameters as training
    MODEL_DIM = 256
    N_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    EMBEDDING_DIM = 16
    kinematic_features = 6
    combined_player_features = kinematic_features + EMBEDDING_DIM
    model_input_dim = 22 * combined_player_features

    model = OracleTransformer(
        input_dim=model_input_dim, model_dim=MODEL_DIM, nhead=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        num_players_total=NUM_UNIQUE_PLAYERS + 1, embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- 4. Run Inference Loop ---
    print("\n--- Starting Inference ---")
    results = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating Predictions"):
            src = batch['X'].to(DEVICE)
            tgt = batch['y'].to(DEVICE)
            player_ids = batch['player_ids'].to(DEVICE)

            prediction = model(src, player_ids, tgt)

            # For each play in the batch, save the results and the player order
            for i in range(src.shape[0]):
                game_id = batch['game_id'][i].item()
                play_id = batch['play_id'][i].item()

                # Get the player order for this specific play
                play_df = full_tracking_df[(full_tracking_df['game_id'] == game_id) & (full_tracking_df['play_id'] == play_id)]
                play_df_clean = play_df.drop_duplicates(subset=['frame_id', 'nfl_id'])
                player_order = list(play_df_clean.pivot(index='frame_id', columns='nfl_id', values='x').columns)

                results.append({
                    'game_id': game_id,
                    'play_id': play_id,
                    'player_order': json.dumps(player_order),
                    'actual_path': json.dumps(tgt[i].cpu().numpy().tolist()),
                    'predicted_path': json.dumps(prediction[i].cpu().numpy().tolist()),
                    'input_path': json.dumps(src[i].cpu().numpy().tolist())
                })

    # --- 5. Save Results ---
    results_df = pd.DataFrame(results)
    output_path = RESULTS_SAVE_DIR / "inference_results.parquet"
    results_df.to_parquet(output_path, index=False)

    print("\n--- Inference Complete ---")
    print(f"Results for {len(results_df)} plays, including player order, saved to: {output_path}")

if __name__ == "__main__":
    main()
