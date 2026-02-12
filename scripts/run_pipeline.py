import pandas as pd
import hockey_elo_utils as elo
import os
import sys

# Ensure we can import local modules if running from outside scripts dir
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Configuration
BASE_DIR = os.path.dirname(script_dir) # d:/A/Warton
DATA_DIR = os.path.join(BASE_DIR, 'Data')

DATA_FILE = os.path.join(DATA_DIR, 'whl_2025.csv')
MATCHUPS_FILE = os.path.join(DATA_DIR, 'matchups.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'submission.csv')

def main():
    print("🚀 Starting Hockey Elo Pipeline...")
    print(f"   Data File: {DATA_FILE}")
    
    # 1. Load and Clean
    df_raw = elo.load_and_clean(DATA_FILE)
    if df_raw.empty:
        print("❌ Data load failed. Stopping.")
        return

    # 2. Aggregate
    df_games = elo.aggregate_to_games(df_raw)
    
    # 3. Restructure (Long Format)
    df_long = elo.create_long_format(df_games)
    
    # 4. Feature Engineering (Multipliers & Perf Score)
    df_train = elo.calculate_multipliers(df_long)
    
    # 5. Run Elo Simulation
    # Using parameters from the notebook: K=30, HFA=14.89
    df_elo = elo.run_elo_simulation(
        df_train, 
        n_sims=1000, 
        k_factor=30, 
        hfa=14.89
    )
    
    # 6. Predict Matchups
    if os.path.exists(MATCHUPS_FILE):
        print(f"\n🔮 Predicting matchups from {MATCHUPS_FILE}...")
        df_preds = elo.predict_matchups(MATCHUPS_FILE, df_elo)
        
        if not df_preds.empty:
            print("\nPrediction Table:")
            print(df_preds[['Home', 'Away', 'Prob_Home', 'Predicted_Winner', 'Risk_Level']].head(10))
            
            # Save
            df_preds.to_csv(SUBMISSION_FILE, index=False)
            print(f"\n✅ Predictions saved to {SUBMISSION_FILE}")
        else:
            print("⚠️ No predictions generated (empty result).")
    else:
        print(f"⚠️ Matchups file not found at {MATCHUPS_FILE}")

if __name__ == "__main__":
    main()
