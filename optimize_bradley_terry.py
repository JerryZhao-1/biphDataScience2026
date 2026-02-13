import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

def negative_log_likelihood(params, home_indices, away_indices, outcome_indices, num_teams):
    """
    Calculates the negative log likelihood of the data given the parameters.
    
    Params:
    -------
    params : array-like
        Array containing [log_pi_0, ..., log_pi_{N-1}, log_nu, log_gamma]
    home_indices : array-like
        Indices of home teams for each game
    away_indices : array-like
        Indices of away teams for each game
    outcome_indices : array-like
        Indices representing the outcome (0=RW, 1=OW, 2=OL, 3=RL)
    num_teams : int
        Number of teams (N)
        
    Returns:
    --------
    nll : float
        Negative Log Likelihood
    """
    # 1. Unpack parameters
    # The first num_teams parameters are log(pi)
    log_pi = params[:num_teams]
    log_nu = params[num_teams]
    log_gamma = params[num_teams + 1]
    
    # Convert to actual values (exponentiate) to ensure positivity
    pi = np.exp(log_pi)
    nu = np.exp(log_nu)
    gamma = np.exp(log_gamma)
    
    # 2. Get strengths for each game
    pi_h = pi[home_indices]
    pi_a = pi[away_indices]
    
    # 3. Calculate terms for probabilities (Vectorized for performance)
    # P(RW) numerator term
    term_rw = gamma * pi_h
    
    # Common factors to avoid recomputing
    # (gamma * pi_i)^(2/3)
    term_rw_23 = term_rw ** (2/3)
    # (gamma * pi_i)^(1/3)
    term_rw_13 = term_rw ** (1/3)
    
    # pi_j^(1/3) and pi_j^(2/3)
    pi_a_13 = pi_a ** (1/3)
    pi_a_23 = pi_a ** (2/3)
    
    # Calculate unnormalized probabilities (the terms in the formula)
    # Note: These are NOT probabilities yet, just terms in the numerator/denominator
    # Formula components:
    # 1. gamma * pi_i
    val_rw = term_rw
    # 2. nu * (gamma * pi_i)^(2/3) * pi_j^(1/3)
    val_ow = nu * term_rw_23 * pi_a_13
    # 3. nu * (gamma * pi_i)^(1/3) * pi_j^(2/3)
    val_ol = nu * term_rw_13 * pi_a_23
    # 4. pi_j
    val_rl = pi_a
    
    denominator = val_rw + val_ow + val_ol + val_rl
    
    # 4. Calculate probabilities for the *observed* outcome only
    # We lay out the values in a (N_games, 4) matrix
    probs_matrix = np.column_stack([val_rw, val_ow, val_ol, val_rl])
    
    # Normalize by denominator
    probs_matrix = probs_matrix / denominator[:, np.newaxis]
    
    # 5. Select the probability of the actual outcome for each game
    # outcome_indices corresponds to columns: 0=RW, 1=OW, 2=OL, 3=RL
    observed_probs = probs_matrix[np.arange(len(outcome_indices)), outcome_indices]
    
    # 6. Sum of log probabilities
    # Add a small epsilon to prevent log(0) if something goes wrong
    epsilon = 1e-15
    log_likelihood = np.sum(np.log(observed_probs + epsilon))
    
    return -log_likelihood

def optimize_team_strengths(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check if we have data
    if df.empty:
        print("Error: DataFrame is empty.")
        return

    # 1. Map Teams to Indices
    # Get sorted list of all unique teams from both columns to ensure consistency
    teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    num_teams = len(teams)
    print(f"Found {num_teams} teams.")
    
    # Create index arrays
    df['home_idx'] = df['home_team'].map(team_to_idx)
    df['away_idx'] = df['away_team'].map(team_to_idx)
    
    # Map Outcomes to Indices
    # RW=0, OW=1, OL=2, RL=3
    outcome_map = {'RW': 0, 'OW': 1, 'OL': 2, 'RL': 3, 'T': 3} # treating Tie as RL for now or handle appropriately
    # Note: If 'T' exists, we might need to handle it. Assuming clean RW/OW/OL/RL for now.
    
    # Check for unexpected outcomes
    unexpected = df[~df['game_outcome'].isin(outcome_map.keys())]['game_outcome'].unique()
    if len(unexpected) > 0:
        print(f"Warning: Found unexpected outcomes: {unexpected}. These will fail mapping.")
        
    df['outcome_idx'] = df['game_outcome'].map(outcome_map)
    
    # Prepare data for optimization
    home_indices = df['home_idx'].values.astype(int)
    away_indices = df['away_idx'].values.astype(int)
    outcome_indices = df['outcome_idx'].values.astype(int)
    
    # 2. Initialize Parameters
    # num_teams for log_pi, 1 for log_nu, 1 for log_gamma
    # Random initialization for stability (close to 0 => pi=1)
    np.random.seed(42)
    initial_log_pi = np.random.normal(0, 0.1, num_teams) 
    initial_log_nu = np.log(1.0) # Start nu at 1.0 (log(1)=0)
    initial_log_gamma = np.log(1.0) # Start gamma at 1.0
    
    initial_params = np.concatenate([initial_log_pi, [initial_log_nu, initial_log_gamma]])
    
    # 3. Define Constraints
    # Constraint: Sum of log_pi = 0 (Mean log strength = 0)
    # logical indexing: params[:-2] are the log_pi values
    def constraint_func(params):
        return np.sum(params[:num_teams])
        
    constraints = ({'type': 'eq', 'fun': constraint_func})
    
    # 4. Run Optimization
    print("Starting optimization...")
    # Using 'SLSQP' as it handles equality constraints well
    result = minimize(
        negative_log_likelihood, 
        initial_params, 
        args=(home_indices, away_indices, outcome_indices, num_teams),
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000, 'disp': True}
    )
    
    print("\nOptimization Result:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final Negative Log Likelihood: {result.fun:.4f}")
    
    # 5. Extract Results
    optimized_params = result.x
    log_pi_opt = optimized_params[:num_teams]
    log_nu_opt = optimized_params[num_teams]
    log_gamma_opt = optimized_params[num_teams + 1]
    
    # Convert back to regular scale
    pi_opt = np.exp(log_pi_opt)
    nu_opt = np.exp(log_nu_opt)
    gamma_opt = np.exp(log_gamma_opt)
    
    print(f"\nOptimization Parameters:")
    print(f"Overtime Parameter (nu): {nu_opt:.4f}")
    print(f"Home Advantage (gamma): {gamma_opt:.4f}")
    
    # Create DataFrame of results
    results_df = pd.DataFrame({
        'Team': teams,
        'Strength': pi_opt,
        'Log_Strength': log_pi_opt
    })
    
    # Sort by strength (descending)
    results_df = results_df.sort_values('Strength', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 Teams by Strength:")
    print(results_df.head(10))
    
    if output_path:
        print(f"\nSaving team strengths to {output_path}...")
        results_df.to_csv(output_path, index=False)
        
    # Save global parameters
    import json
    params_file = os.path.join(os.path.dirname(output_path), "model_params.json")
    with open(params_file, 'w') as f:
        json.dump({'nu': nu_opt, 'gamma': gamma_opt}, f)
    print(f"Model parameters saved to {params_file}")

    return results_df, nu_opt, gamma_opt

if __name__ == "__main__":
    base_dir = r"d:\A\Warton\Data"
    input_file = os.path.join(base_dir, "whl_2025_games.csv")
    output_file = os.path.join(base_dir, "whl_team_strengths.csv")
    
    optimize_team_strengths(input_file, output_file)
