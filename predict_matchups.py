import pandas as pd
import json
import os
from bradley_terry_likelihood import calculate_probabilities

def predict_matchups():
    data_dir = r"d:\A\Warton\Data"
    matchups_file = os.path.join(data_dir, "matchups.csv")
    strengths_file = os.path.join(data_dir, "whl_power_rankings.csv")
    params_file = os.path.join(data_dir, "model_params.json")
    
    # 1. Load Data
    print("Starting prediction script...")
    try:
        if not os.path.exists(matchups_file): print(f"Missing {matchups_file}")
        if not os.path.exists(strengths_file): print(f"Missing {strengths_file}")
        if not os.path.exists(params_file): print(f"Missing {params_file}")
        
        matchups_df = pd.read_csv(matchups_file)
        strengths_df = pd.read_csv(strengths_file)
        with open(params_file, 'r') as f:
            params = json.load(f)
            nu = params['nu']
            gamma = params['gamma']
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Create a dictionary for quick strength lookup
    # Normalize team names to lower case for consistent matching
    strengths_df['Team_Lower'] = strengths_df['Team'].str.lower().str.strip()
    team_strength_map = dict(zip(strengths_df['Team_Lower'], strengths_df['Strength_Pi']))
    
    print(f"Loaded {len(matchups_df)} matchups.")
    print(f"Global Parameters: nu={nu:.4f}, gamma={gamma:.4f}")
    print("\n" + "="*80)
    print(f"{'Matchup':<40} | {'Home Win Prob (RW+OW)':<25}")
    print("="*80)
    
    results = []
    
    for _, row in matchups_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get strengths
        pi_home = team_strength_map.get(str(home_team).lower().strip())
        pi_away = team_strength_map.get(str(away_team).lower().strip())
        
        if pi_home is None or pi_away is None:
            print(f"Error: Could not find strength for {home_team} or {away_team}")
            continue
            
        # Calculate Probabilities
        prob_rw, prob_ow, prob_ol, prob_rl = calculate_probabilities(pi_home, pi_away, nu, gamma)
        
        # Total Home Win Probability (Regulation + Overtime)
        home_win_prob = prob_rw + prob_ow
        
        # Percentage
        win_pct = home_win_prob * 100
        
        print(f"{home_team} vs {away_team:<25} | {win_pct:.1f}%")
        
        results.append({
            'Home': home_team,
            'Away': away_team,
            'Home_Win_Prob': home_win_prob,
            'RW_Prob': prob_rw,
            'OW_Prob': prob_ow,
            'OL_Prob': prob_ol,
            'RL_Prob': prob_rl
        })

    print("="*80)
    
    # Save detailed predictions
    output_df = pd.DataFrame(results)
    output_path = os.path.join(data_dir, "matchup_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nDetailed predictions saved to {output_path}")

if __name__ == "__main__":
    predict_matchups()
