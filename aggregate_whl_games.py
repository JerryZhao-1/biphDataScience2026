import pandas as pd
import os

def aggregate_games(input_path, output_path=None):
    """
    Aggregates WHL 2025 shift-level data to game-level data.
    """
    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None

    # Verify required columns exist
    required_columns = ['game_id', 'home_team', 'away_team', 'went_ot', 'home_goals', 'away_goals']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Dataset missing one of the required columns: {required_columns}")
        return None

    print("Aggregating data...")
    # 1. Group by game_id
    # 2. Keep first instance of constant columns, sum goals
    # Note: reset_index() makes game_id a column again
    games_df = df.groupby('game_id').agg({
        'home_team': 'first',
        'away_team': 'first',
        'went_ot': 'first',
        'home_goals': 'sum',
        'away_goals': 'sum'
    }).reset_index()

    print("calculating game outcomes...")
    # 4. Create game_outcome column
    # Logic:
    # RW: home_goals > away_goals AND went_ot == 0
    # OW: home_goals > away_goals AND went_ot == 1
    # OL: home_goals < away_goals AND went_ot == 1
    # RL: home_goals < away_goals AND went_ot == 0
    
    def determine_outcome(row):
        h = row['home_goals']
        a = row['away_goals']
        ot = row['went_ot']
        
        if h > a:
            return 'OW' if ot == 1 else 'RW'
        elif a > h:
            return 'OL' if ot == 1 else 'RL'
        else:
            return 'T' # Handle ties if any exist (though rules imply outcomes)

    games_df['game_outcome'] = games_df.apply(determine_outcome, axis=1)

    print("Aggregation complete.")
    print(f"Processed {len(games_df)} games.")
    print("\nSample of games_df:")
    print(games_df.head())

    if output_path:
        print(f"Saving to {output_path}...")
        games_df.to_csv(output_path, index=False)
        print("Done.")
    
    return games_df

if __name__ == "__main__":
    # Define paths
    base_dir = r"d:\A\Warton\Data"
    input_file = os.path.join(base_dir, "whl_2025.csv")
    output_file = os.path.join(base_dir, "whl_2025_games.csv")
    
    # Run aggregation
    aggregate_games(input_file, output_file)
