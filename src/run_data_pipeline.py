# --- NFL Big Data Bowl 2026: Project Oracle ---
# Phase 1.3.4: Main Data Processing Pipeline
#
# This script orchestrates the entire data cleaning and preparation process.
# It takes the raw weekly CSV files, applies the functions from data_processing.py,
# and saves the final, clean dataset for each week as a .parquet file.

import pandas as pd
from pathlib import Path
from tqdm import tqdm # A library to show progress bars for long operations

# Import the functions we built from our other script
from data_processing import (
    merge_supplementary_data,
    filter_plays,
    standardize_coordinates
)

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("--- Starting Project Oracle Data Processing Pipeline ---")

    # --- 1. Define File Paths ---
    # Use pathlib for robust path management
    RAW_DATA_DIR = Path("./data/raw/")
    PROCESSED_DATA_DIR = Path("./data/processed/")

    # Create the processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    supplementary_path = RAW_DATA_DIR / "supplementary_data.csv"
    train_dir = RAW_DATA_DIR / "train"

    # --- 2. Load the Supplementary Data ---
    # This file is loaded once as it applies to all weekly files.
    print(f"Loading supplementary data from: {supplementary_path}")
    supplementary_df = pd.read_csv(supplementary_path)
    print("Supplementary data loaded successfully.")

    # --- 3. Find All Weekly Input Files ---
    # We will loop through each of these and process them one by one.
    weekly_input_files = sorted(list(train_dir.glob("input_*.csv")))
    print(f"Found {len(weekly_input_files)} weekly input files to process.")

    # --- 4. Process Each Weekly File ---
    for input_path in tqdm(weekly_input_files, desc="Processing weekly files"):
        week_str = input_path.stem.split('_')[-1] # Extracts 'w01', 'w02', etc.
        print(f"\nProcessing {input_path.name}...")

        # Load the raw weekly tracking data
        tracking_df = pd.read_csv(input_path)

        # Step A: Merge with supplementary data
        print("  Step A: Merging supplementary data...")
        merged_df = merge_supplementary_data(tracking_df, supplementary_df)

        # Step B: Filter out invalid plays
        print("  Step B: Filtering invalid plays...")
        filtered_df = filter_plays(merged_df)

        # Step C: Standardize coordinates for every play
        # This is the most computationally intensive step.
        # We group by each unique play and apply our function to each group.
        print("  Step C: Standardizing coordinates (this may take a moment)...")
        # Using .copy() to ensure the original filtered_df is not modified,
        # which can be good practice with groupby().apply().
        standardized_df = filtered_df.groupby(['game_id', 'play_id'], group_keys=False, include_groups=False).apply(standardize_coordinates).copy()

        # --- 5. Save the Processed File ---
        output_filename = f"processed_{week_str}.parquet"
        output_path = PROCESSED_DATA_DIR / output_filename

        print(f"  Saving processed data to: {output_path}")
        # Using .parquet format is more efficient for storage and reading than CSV
        standardized_df.to_parquet(output_path)
        print(f"Successfully processed and saved {input_path.name}.")

    print("\n--- Data Processing Pipeline Complete ---")
    print(f"All processed files have been saved to the '{PROCESSED_DATA_DIR}' directory.")


if __name__ == "__main__":
    main()
