import json
import os

def create_notebook():
    notebook_content = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(source):
        notebook_content["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        notebook_content["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # --- Notebook Content Construction ---

    # Title
    add_markdown("# Bradley-Terry Model for WHL 2025\n\nScale-based implementation following Whelan & Klein (Four-Outcome Model).")

    # Imports
    add_code("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Global Configuration
DATA_DIR = r"d:\\A\\Warton\\Data"
INPUT_FILE = os.path.join(DATA_DIR, "whl_2025.csv")
MATCHUPS_FILE = os.path.join(DATA_DIR, "matchups.csv")
""")

    # Phase 1
    add_markdown("## Phase 1: Data Pre-processing & Aggregation\n\nAggregate shift-level data to game-level data with Regulation/Overtime outcomes.")
    add_code("""
def load_and_aggregate_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Aggregation Logic
    # 1. Group by game_id
    games_df = df.groupby('game_id').agg({
        'home_team': 'first',
        'away_team': 'first',
        'went_ot': 'first',
        'home_goals': 'sum',
        'away_goals': 'sum',
        'home_xg': 'sum',  # Aggregating xG for potential use
        'away_xg': 'sum'
    }).reset_index()
    
    # 4. Determine Outcome (Whelan & Klein Four-Outcome logic)
    def get_outcome(row):
        h, a, ot = row['home_goals'], row['away_goals'], row['went_ot']
        if h > a:
            return 'OW' if ot == 1 else 'RW'
        else: # a > h (Assuming no ties in final result)
            return 'OL' if ot == 1 else 'RL' # OL from Home perspective
            
    games_df['game_outcome'] = games_df.apply(get_outcome, axis=1)
    
    print(f"Aggregated {len(games_df)} games.")
    return df, games_df # Return raw and aggregated

# Execute Phase 1
df_raw, games_df = load_and_aggregate_data(INPUT_FILE)
games_df.head()
""")

    # Phase 2
    add_markdown("## Phase 2: Defining the Mathematical Model\n\nImplement the Whelan & Klein probability formulas with Home Advantage (gamma) and Overtime Tendency (nu).")
    add_code("""
def calculate_probabilities(pi_home, pi_away, nu, gamma):
    \"\"\"
    Calculates P(RW), P(OW), P(OL), P(RL) for Home team.
    \"\"\"
    # Ensure inputs are arrays/floats
    pi_i = np.asarray(pi_home, dtype=float)
    pi_j = np.asarray(pi_away, dtype=float)
    
    # Derived terms
    # Using 2/3 and 1/3 exponents based on points system (3-2-1-0)
    term_rw = gamma * pi_i
    term_ow = nu * (term_rw)**(2/3) * (pi_j)**(1/3)
    term_ol = nu * (term_rw)**(1/3) * (pi_j)**(2/3)
    term_rl = pi_j
    
    denominator = term_rw + term_ow + term_ol + term_rl
    
    return (
        term_rw / denominator,
        term_ow / denominator,
        term_ol / denominator,
        term_rl / denominator
    )
""")

    # Phase 3
    add_markdown("## Phase 3: Model Estimation\n\nMaximize Log-Likelihood to find optimal Team Strengths (pi).")
    add_code("""
def optimize_model(games_df):
    # 1. Map Teams to Indices
    teams = sorted(list(set(games_df['home_team']) | set(games_df['away_team'])))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    
    # Prepare Arrays for Vectorized Likelihood
    home_idx = games_df['home_team'].map(team_to_idx).values
    away_idx = games_df['away_team'].map(team_to_idx).values
    
    outcome_map = {'RW': 0, 'OW': 1, 'OL': 2, 'RL': 3}
    outcomes = games_df['game_outcome'].map(outcome_map).values
    
    # Negative Log Likelihood Function
    def neg_log_likelihood(params):
        # Unpack parameters: [log_pi (n_teams), log_nu, log_gamma]
        log_pi = params[:n_teams]
        nu = np.exp(params[n_teams])
        gamma = np.exp(params[n_teams+1])
        pi = np.exp(log_pi)
        
        pi_h = pi[home_idx]
        pi_a = pi[away_idx]
        
        # Calculate Probabilities for all games
        probs_matrix = np.column_stack(calculate_probabilities(pi_h, pi_a, nu, gamma))
        
        # Select prob of observed outcome
        observed_probs = probs_matrix[np.arange(len(outcomes)), outcomes]
        
        # return negative sum of log probs
        return -np.sum(np.log(observed_probs + 1e-15))
        
    # Constraints: Sum of log_pi = 0 (Geometric mean of pi = 1)
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p[:n_teams])}]
    
    # Initialization
    x0 = np.concatenate([np.random.normal(0, 0.1, n_teams), [0.0, 0.0]]) # log(1)=0 for nu/gamma
    
    print("Optimizing...")
    res = minimize(neg_log_likelihood, x0, constraints=constraints, method='SLSQP')
    
    print(f"Optimization Success: {res.success}")
    
    # Extract Results
    pi_opt = np.exp(res.x[:n_teams])
    nu_opt = np.exp(res.x[n_teams])
    gamma_opt = np.exp(res.x[n_teams+1])
    
    print(f"nu (Overtime): {nu_opt:.4f}")
    print(f"gamma (Home Adv): {gamma_opt:.4f}")
    
    return teams, pi_opt, nu_opt, gamma_opt

teams, pi_opt, nu_opt, gamma_opt = optimize_model(games_df)
""")

    # Phase 4
    add_markdown("## Phase 4: Power Rankings\n\nTop 10 Teams by Underlying Strength.")
    add_code("""
# Create DataFrame
rankings = pd.DataFrame({'Team': teams, 'Strength_Pi': pi_opt})
rankings = rankings.sort_values('Strength_Pi', ascending=False).reset_index(drop=True)
rankings['Rank'] = rankings.index + 1

print("Top 10 Teams:")
print(rankings.head(10)[['Rank', 'Team', 'Strength_Pi']])

# Save
rankings.to_csv(os.path.join(DATA_DIR, "whl_power_rankings.csv"), index=False)
""")

    # Phase 5
    add_markdown("## Phase 5: Tournament Predictions\n\nPredict Home Win Probability (RW + OW) for specified matchups.")
    add_code("""
def predict_matchups(matchups_file, rankings, nu, gamma):
    if not os.path.exists(matchups_file):
        print("Matchups file not found.")
        return
        
    matchups = pd.read_csv(matchups_file)
    team_map = dict(zip(rankings['Team'], rankings['Strength_Pi']))
    
    print(f"{'Matchup':<40} | {'Home Win Prob':<15}")
    print("-" * 60)
    
    predictions = []
    
    for _, row in matchups.iterrows():
        h, a = row['home_team'], row['away_team']
        if h not in team_map or a not in team_map:
            print(f"Skipping {h} vs {a} (Team not found)")
            continue
            
        pi_h = team_map[h]
        pi_a = team_map[a]
        
        probs = calculate_probabilities(pi_h, pi_a, nu, gamma)
        # Home Win = RW + OW (Outcome indices 0 and 1)
        p_win = probs[0] + probs[1]
        
        print(f"{h} vs {a:<25} | {p_win*100:.1f}%")
        predictions.append({'Home': h, 'Away': a, 'Home_Win_Prob': p_win})
        
    pd.DataFrame(predictions).to_csv(os.path.join(DATA_DIR, "matchup_predictions.csv"), index=False)

predict_matchups(MATCHUPS_FILE, rankings, nu_opt, gamma_opt)
""")

    # Phase 6
    add_markdown("## Phase 6: Model Validation (Train/Test Split)\n\nSplit data 80/20, retrain model, and evaluate prediction accuracy.")
    add_code("""
# 1. Split Data
train_df, test_df = train_test_split(games_df, test_size=0.2, random_state=42)
print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# 2. Train Model on 80% Data
print("\\nTraining model on 80% data...")
teams_train, pi_train, nu_train, gamma_train = optimize_model(train_df)

# Map teams to strength
team_map_train = dict(zip(teams_train, pi_train))

# 3. Predict on 20% Data
correct_predictions = 0
valid_test_samples = 0

print("\\nEvaluating on test set (Win vs Loss)...")
for _, row in test_df.iterrows():
    h, a = row['home_team'], row['away_team']
    actual_outcome = row['game_outcome']
    
    # Skip if team not seen in training
    if h not in team_map_train or a not in team_map_train:
        continue
        
    valid_test_samples += 1
    
    # Determine Actual Result (Win vs Loss) for Home Team
    if actual_outcome in ['RW', 'OW']:
        actual_result = 'Win'
    else:
        actual_result = 'Loss'
    
    pi_h = team_map_train[h]
    pi_a = team_map_train[a]
    
    # Calculate outcome probabilities
    probs = calculate_probabilities(pi_h, pi_a, nu_train, gamma_train)
    
    # Probability of Home Win (RW + OW)
    prob_win = probs[0] + probs[1]
    
    # If P(Win) > 0.5, predict Win
    if prob_win > 0.5:
        pred_result = 'Win'
    else:
        pred_result = 'Loss'
    
    if pred_result == actual_result:
        correct_predictions += 1

# 4. Output Accuracy
if valid_test_samples > 0:
    accuracy = correct_predictions / valid_test_samples
    print(f"\\nTest Set Accuracy (Win/Loss): {accuracy:.2%}")
    print(f"Correct Predictions: {correct_predictions}/{valid_test_samples}")
else:
    print("No valid test samples found (teams missing from training set).")
""")

    # Save Notebook
    output_path = r"d:\A\Warton\JupyterNotebook\Bradley-Terry.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=1)
    print("Write successful.")
    
    print(f"Notebook updated at {output_path}")

if __name__ == "__main__":
    create_notebook()
