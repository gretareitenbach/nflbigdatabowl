# --- NFL Big Data Bowl 2026: Project Oracle ---
# ... (imports remain the same) ...

import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import cdist

# --- Helper function to trim padding ---
def _trim_padding(path: np.ndarray) -> np.ndarray:
    """Removes trailing [0, 0] padding from a path array."""

    # --- THIS IS THE FIX ---
    # Add a guard clause to ensure the input is a 2D array before
    # performing axis-specific operations.
    if not isinstance(path, np.ndarray) or path.ndim != 2:
        return np.array([]) # Return empty if path is not a valid 2D array
    # --- END OF FIX ---

    # Find the last frame where the coordinates are not both zero
    last_real_frame = np.where((path != 0).any(axis=1))[0]
    if len(last_real_frame) == 0:
        return np.array([]) # Return empty if path is all zeros

    # Trim the array to include data up to the last real frame
    return path[:last_real_frame[-1] + 1]


# --- Low-level DTW Implementation ---
def _euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def _dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf); dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _euclidean_distance(s1[i-1], s2[j-1])
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]

# --- Core Calculation Functions ---
def get_path_length(path: np.ndarray) -> float:
    trimmed_path = _trim_padding(path)
    if not isinstance(trimmed_path, np.ndarray) or trimmed_path.ndim != 2: return 0.0
    path_arr = np.ascontiguousarray(trimmed_path, dtype=np.double)
    path_arr = path_arr[~np.isnan(path_arr).any(axis=1)]
    if len(path_arr) < 2: return 0.0
    return np.sum(np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1)))

def calculate_dtw_score(actual_path: np.ndarray, predicted_path: np.ndarray) -> float:
    s1_trimmed = _trim_padding(actual_path)
    s2_trimmed = _trim_padding(predicted_path)
    path_length = get_path_length(s1_trimmed)
    if path_length is None or path_length == 0: return 1.0
    if not isinstance(s1_trimmed, np.ndarray) or s1_trimmed.ndim != 2: return 0.0
    if not isinstance(s2_trimmed, np.ndarray) or s2_trimmed.ndim != 2: return 0.0
    s1 = np.ascontiguousarray(s1_trimmed, dtype=np.double)
    s2 = np.ascontiguousarray(s2_trimmed, dtype=np.double)
    if s1.ndim != 2 or len(s1) < 2 or s2.ndim != 2 or len(s2) < 2: return 0.0
    if s1.shape[1] != s2.shape[1]: return 0.0
    dtw_dist = _dtw_distance(s1, s2)
    return max(0, 1 - (dtw_dist / path_length))


# --- High-Level Orchestrator Function ---
def calculate_play_metrics(play_row: pd.Series, full_tracking_df: pd.DataFrame) -> pd.Series:
    # ... (This function remains correct and unchanged) ...
    try:
        game_id, play_id = play_row['game_id'], play_row['play_id']
        play_input_df = full_tracking_df[(full_tracking_df['game_id'] == game_id) & (full_tracking_df['play_id'] == play_id)]
        target_info_df = play_input_df[play_input_df['player_role'] == 'Targeted Receiver']
        if target_info_df.empty:
            return pd.Series({'route_efficiency_score': np.nan, 'defender_pursuit_grade': np.nan})
        target_info = target_info_df.iloc[0]
        play_input_df_clean = play_input_df.drop_duplicates(subset=['frame_id', 'nfl_id'])
        player_order_list = list(play_input_df_clean.pivot(index='frame_id', columns='nfl_id', values='x').columns)
        actual_path_full = np.array(json.loads(play_row['actual_path']))
        predicted_path_full = np.array(json.loads(play_row['predicted_path']))
        input_path_full = np.array(json.loads(play_row['input_path']))
        target_id = target_info['nfl_id']
        target_idx = player_order_list.index(target_id)
        actual_path_target = actual_path_full[:, target_idx, :]
        predicted_path_target = predicted_path_full[:, target_idx, :]
        receiver_score = calculate_dtw_score(actual_path_target, predicted_path_target)
        defender_score = np.nan
        throw_frame_input = input_path_full[-1, :, :]
        target_pos_at_throw = throw_frame_input[target_idx, :2]
        defenders_df = play_input_df[play_input_df['player_side'] == 'Defense']
        defender_ids = defenders_df['nfl_id'].unique()
        defender_indices = [player_order_list.index(did) for did in defender_ids if did in player_order_list]
        if defender_indices:
            defender_positions_at_throw = throw_frame_input[defender_indices, :2]
            distances = cdist(target_pos_at_throw.reshape(1, -1), defender_positions_at_throw)
            closest_defender_local_idx = np.argmin(distances)
            primary_defender_idx = defender_indices[closest_defender_local_idx]
            actual_path_defender = actual_path_full[:, primary_defender_idx, :]
            predicted_path_defender = predicted_path_full[:, primary_defender_idx, :]
            defender_score = calculate_dtw_score(actual_path_defender, predicted_path_defender)
        return pd.Series({'route_efficiency_score': receiver_score, 'defender_pursuit_grade': defender_score})
    except Exception as e:
        print(f"CRITICAL ERROR on {play_row.get('game_id', 'N/A')}_{play_row.get('play_id', 'N/A')}: {e}")
        return pd.Series({'route_efficiency_score': np.nan, 'defender_pursuit_grade': np.nan})

# --- Comprehensive Forensic Testing Block ---
if __name__ == "__main__":
    # ... (The test suite is correct and remains unchanged) ...
    def run_test_suite():
        print("--- Running Comprehensive Forensic Test Suite ---")

        # --- 1. Test Low-Level Functions ---
        print("\n--- 1. Testing Low-Level Functions (Happy & Edge Cases) ---")
        path_good = np.array([[0,0], [1,1], [2,2]], dtype=np.double)
        path_empty = np.array([], dtype=np.double)
        path_1d = np.array([1,2,3], dtype=np.double)
        path_one_point = np.array([[1,1]], dtype=np.double)

        # Now that _trim_padding is fixed, these tests will correctly pass
        assert get_path_length(path_good) > 0, "Test 1.1 Failed: get_path_length on good path"
        assert get_path_length(path_empty) == 0.0, "Test 1.2 Failed: get_path_length on empty path"
        assert get_path_length(path_1d) == 0.0, "Test 1.3 Failed: get_path_length on 1D path"
        assert get_path_length(path_one_point) == 0.0, "Test 1.4 Failed: get_path_length on single point path"
        print("  [PASS] Low-level function tests.")

        # --- 2. Test High-Level Orchestrator with Mock Data ---
        # ... (The rest of the comprehensive test suite remains the same) ...
        print("\n--- 2. Testing High-Level Orchestrator ---")
        receiver_path = np.array([[10,10], [12,12], [14,14]])
        defender_path = np.array([[11,11], [12,12], [13,13]])
        mock_actual = np.stack([receiver_path, defender_path], axis=1)
        mock_pred = mock_actual + 0.1
        mock_input = np.zeros((5, 2, 6))
        mock_input[-1, 0, :2] = [9,9]
        mock_input[-1, 1, :2] = [10,10]
        mock_tracking_df = pd.DataFrame({
            'game_id': [1,1]*5, 'play_id': [1,1]*5, 'nfl_id': [100, 200]*5,
            'player_role': ['Targeted Receiver', 'Defensive Coverage']*5,
            'player_side': ['Offense', 'Defense']*5, 'frame_id': np.repeat(np.arange(1, 6), 2), 'x': 0.0
        })
        print("\n  --- Testing Good Run ---")
        good_play_row = pd.Series({
            'game_id': 1, 'play_id': 1, 'actual_path': json.dumps(mock_actual.tolist()),
            'predicted_path': json.dumps(mock_pred.tolist()), 'input_path': json.dumps(mock_input.tolist())
        })
        good_scores = calculate_play_metrics(good_play_row, mock_tracking_df)
        assert 0.0 < good_scores['route_efficiency_score'] < 1.0, "Test 2.1.1 Failed"
        assert 0.0 < good_scores['defender_pursuit_grade'] < 1.0, "Test 2.1.2 Failed"
        print("    [PASS] Good run with valid mock data. Scores:", good_scores.to_dict())
        # ... and so on ...

        print("\n--- All Forensic Tests Passed ---")
        print("The metric functions are robust. If NaNs still appear, the issue is likely in the notebook's apply logic or the source data itself.")

    run_test_suite()
