# --- NFL Big Data Bowl 2026: Project Oracle ---
# Phase 1.3: Data Processing
#
# This script contains reusable functions for cleaning and preparing the competition data.
# - standardize_coordinates: Orients all plays to move left-to-right.
# - merge_supplementary_data: Joins play-level context onto tracking data.
# - filter_plays: Selects only valid, standard passing plays for analysis.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def standardize_coordinates(play_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes player tracking data so that all plays appear to move from left to right.
    """
    # ... (code from previous version, no changes) ...
    df = play_df.copy()
    direction = df['play_direction'].iloc[0]
    if direction == 'left':
        df['x'] = 120.0 - df['x']
        df['y'] = 160.0/3.0 - df['y']
        df['o'] = (df['o'] + 180) % 360
        df['dir'] = (df['dir'] + 180) % 360
        df['play_direction'] = 'right'
    return df

def merge_supplementary_data(tracking_df: pd.DataFrame, supplementary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges supplementary play-level data onto a tracking data DataFrame.
    """
    # ... (code from previous version, no changes) ...
    context_cols = [
        'game_id', 'play_id', 'down', 'yards_to_go', 'possession_team',
        'defensive_team', 'offense_formation', 'team_coverage_type',
        'pass_result', 'play_nullified_by_penalty', 'route_of_target'
    ]
    cols_to_use = [col for col in context_cols if col in supplementary_df.columns]
    merged_df = pd.merge(
        tracking_df,
        supplementary_df[cols_to_use],
        how='left',
        on=['game_id', 'play_id']
    )
    return merged_df

def filter_plays(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the merged DataFrame to include only standard, non-penalty passing plays.

    Args:
        merged_df (pd.DataFrame): The DataFrame with both tracking and supplementary data.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the plays to be used for modeling.
    """
    # Define valid pass results. We exclude sacks ('S') and scrambles ('R').
    valid_pass_results = ['C', 'I', 'IN']

    # Apply the filters
    filtered_df = merged_df[
        (merged_df['play_nullified_by_penalty'] == 'N') &
        (merged_df['pass_result'].isin(valid_pass_results))
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    return filtered_df

# --- Testing Block ---
if __name__ == "__main__":

    def create_football_field(ax, title=""):
        ax.add_patch(patches.Rectangle((0, 0), 120, 53.3, facecolor='#2c5b2c', zorder=0))
        ax.add_patch(patches.Rectangle((0, 0), 10, 53.3, facecolor='#1e3f1e', zorder=0))
        ax.add_patch(patches.Rectangle((110, 0), 10, 53.3, facecolor='#1e3f1e', zorder=0))
        for x in range(20, 110, 10):
            ax.axvline(x, color='white', linestyle='-', linewidth=0.5, zorder=1)
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_title(title)
        ax.axis('off')
        return ax

    print("--- Testing Data Processing Functions ---")

    input_path = Path("C:\\Users\\User\\OneDrive - Massachusetts Institute of Technology\\Personal\\Fun Projects\\nflbigdatabowl\\data\\raw\\train\\input_2023_w01.csv")
    supplementary_path = Path("C:\\Users\\User\\OneDrive - Massachusetts Institute of Technology\\Personal\\Fun Projects\\nflbigdatabowl\\data\\raw\\supplementary_data.csv")

    input_df = pd.read_csv(input_path)
    supplementary_df = pd.read_csv(supplementary_path)

    # --- Test 1: standardize_coordinates ---
    print("\n--- Testing Coordinate Standardization ---")
    # Find the first play that is moving left to use for our test
    left_play_group = input_df[input_df['play_direction'] == 'left'].groupby(['game_id', 'play_id'])
    left_play_id = list(left_play_group.groups.keys())[0]
    play_to_test = input_df.groupby(['game_id', 'play_id']).get_group(left_play_id)

    standardized_play_df = standardize_coordinates(play_to_test)
    print("Standardization function ran. Visual confirmation plot will be saved.")

    # --- Test 2: merge_supplementary_data ---
    print("\n--- Testing Supplementary Data Merge ---")
    merged_test_df = merge_supplementary_data(input_df.head(100), supplementary_df) # Test on a small sample

    print(f"Shape of tracking data sample: {input_df.head(100).shape}")
    print(f"Shape of merged data sample: {merged_test_df.shape}")
    print("Columns added:", [col for col in merged_test_df.columns if col not in input_df.head(100).columns])

    # --- Test 3: filter_plays ---
    print("\n--- Testing Play Filtering ---")
    full_merged_df = merge_supplementary_data(input_df, supplementary_df)
    print(f"Number of rows before filtering: {len(full_merged_df)}")
    filtered_df = filter_plays(full_merged_df)
    print(f"Number of rows after filtering: {len(filtered_df)}")
    print("\nVerifying filter results:")
    print("Unique values in 'play_nullified_by_penalty' after filtering:", filtered_df['play_nullified_by_penalty'].unique())
    print("Unique values in 'pass_result' after filtering:", filtered_df['pass_result'].unique())
    print("\nFilter test complete. Non-standard plays have been removed.")

    # --- Visual Confirmation Plot for Test 1 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1 = create_football_field(axes[0], title="Before Standardization (Direction: Left)")
    for pid in play_to_test['nfl_id'].unique():
        player_data = play_to_test[play_to_test['nfl_id'] == pid]
        ax1.plot(player_data['x'], player_data['y'], color='red' if player_data['player_side'].iloc[0] == 'Defense' else 'blue')

    ax2 = create_football_field(axes[1], title="After Standardization (Direction: Right)")
    for pid in standardized_play_df['nfl_id'].unique():
        player_data = standardized_play_df[standardized_play_df['nfl_id'] == pid]
        ax2.plot(player_data['x'], player_data['y'], color='red' if player_data['player_side'].iloc[0] == 'Defense' else 'blue')

    plt.tight_layout()
    plt.show()
