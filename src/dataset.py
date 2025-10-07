# --- NFL Big Data Bowl 2026: Project Oracle ---
# ... (existing comments) ...

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import numpy as np

class NFLTrackingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading NFL player tracking data.
    Each item in the dataset corresponds to a single play.
    """
    def __init__(self, processed_input_path: Path, raw_output_dir: Path, player_id_map: dict,
                 max_input_frames: int = 70, max_output_frames: int = 50, num_players: int = 22):
        """
        Args:
            processed_input_path (Path): Path to the processed .parquet file (contains input data).
            raw_output_dir (Path): Path to the directory containing the raw output CSV files.
            player_id_map (dict): A dictionary mapping nfl_id to a unique integer index.
            max_input_frames (int): The length to pad/truncate all input sequences to.
            max_output_frames (int): The length to pad/truncate all output sequences to.
            num_players (int): The fixed number of players to ensure each tensor has.
        """
        self.input_data = pd.read_parquet(processed_input_path)

        week_str = processed_input_path.stem.split('_')[-1]
        output_filename = f"output_2023_{week_str}.csv"
        output_path = raw_output_dir / output_filename
        self.output_data = pd.read_csv(output_path)

        self.plays = self.input_data.groupby(['game_id', 'play_id']).groups
        self.play_keys = list(self.plays.keys())

        # --- NEW: Store the player ID map ---
        self.player_id_map = player_id_map

        self.feature_cols = ['x', 'y', 's', 'a', 'o', 'dir']
        self.target_cols = ['x', 'y']

        self.max_input_frames = max_input_frames
        self.max_output_frames = max_output_frames
        self.num_players = num_players

    def __len__(self) -> int:
        return len(self.play_keys)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single play, reshapes it, pads/truncates it, and returns
        the input (X), target (y), and player_ids tensors.
        """
        play_key = self.play_keys[idx]
        game_id, play_id = play_key

        input_play_df = self.input_data[
            (self.input_data['game_id'] == game_id) & (self.input_data['play_id'] == play_id)
        ]
        output_play_df = self.output_data[
            (self.output_data['game_id'] == game_id) & (self.output_data['play_id'] == play_id)
        ]

        # --- Reshape X and get player IDs ---
        actual_num_input_frames = input_play_df['frame_id'].nunique()

        # Pivot to get players as columns
        X_pivot = input_play_df.pivot(index='frame_id', columns='nfl_id', values=self.feature_cols)

        # Get the actual nfl_ids in the order they appear in the pivoted DataFrame
        player_ids_in_play = X_pivot.columns.get_level_values('nfl_id').unique()
        actual_num_players = len(player_ids_in_play)

        X_np = X_pivot.to_numpy().reshape(actual_num_input_frames, actual_num_players, len(self.feature_cols))

        # --- NEW: Create Player ID Tensor ---
        # Map nfl_ids to integer indices
        player_indices = [self.player_id_map.get(pid, 0) for pid in player_ids_in_play] # Default to 0 for unknown
        # Create a tensor that repeats these IDs for each frame
        player_ids_np = np.tile(player_indices, (actual_num_input_frames, 1))

        # --- Reshape y ---
        y_pivot = output_play_df.pivot(index='frame_id', columns='nfl_id', values=self.target_cols)
        actual_num_output_frames = len(y_pivot)
        output_players_in_play = y_pivot.columns.get_level_values('nfl_id').unique()
        y_np = y_pivot.to_numpy().reshape(actual_num_output_frames, len(output_players_in_play), len(self.target_cols))

        # --- Convert to Tensors ---
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        player_ids_tensor = torch.tensor(player_ids_np, dtype=torch.long) # IDs must be long type
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        # --- Padding ---
        padded_X = torch.zeros((self.max_input_frames, self.num_players, len(self.feature_cols)))
        seq_len_X = min(X_tensor.shape[0], self.max_input_frames)
        players_to_copy_X = min(X_tensor.shape[1], self.num_players)
        padded_X[:seq_len_X, :players_to_copy_X, :] = X_tensor[:seq_len_X, :players_to_copy_X, :]

        padded_player_ids = torch.zeros((self.max_input_frames, self.num_players), dtype=torch.long)
        padded_player_ids[:seq_len_X, :players_to_copy_X] = player_ids_tensor[:seq_len_X, :players_to_copy_X]

        # Note: y padding is simplified. A more robust solution would map output players to input player indices.
        padded_y = torch.zeros((self.max_output_frames, self.num_players, len(self.target_cols)))
        seq_len_y = min(y_tensor.shape[0], self.max_output_frames)
        players_to_copy_y = min(y_tensor.shape[1], self.num_players)
        padded_y[:seq_len_y, :players_to_copy_y, :] = y_tensor[:seq_len_y, :players_to_copy_y, :]

        return {
            'X': padded_X,
            'y': padded_y,
            'player_ids': padded_player_ids, # <-- The new, required output
            'game_id': game_id,
            'play_id': play_id
        }

if __name__ == "__main__":
    pass
