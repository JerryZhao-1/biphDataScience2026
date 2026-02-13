import pandas as pd
import numpy as np
import os

def calculate_probabilities(pi_home, pi_away, nu, gamma):
    """
    Calculates the probability of Regulation Win (RW), Overtime Win (OW), 
    Overtime Loss (OL), and Regulation Loss (RL) for the home team
    based on the Whelan & Klein probability formulas.
    
    The formulas are:
    P(RW) = (gamma * pi_i) / D
    P(OW) = (nu * (gamma * pi_i)^(2/3) * pi_j^(1/3)) / D
    P(OL) = (nu * (gamma * pi_i)^(1/3) * pi_j^(2/3)) / D
    P(RL) = pi_j / D
    
    Where D (denominator) is the sum of the four terms above.
    
    Parameters:
    -----------
    pi_home : float or array-like
        Strength of the home team (pi_i)
    pi_away : float or array-like
        Strength of the away team (pi_j)
    nu : float
        Overtime tendency parameter
    gamma : float
        Home advantage parameter applied to home team strength
        
    Returns:
    --------
    tuple of (prob_rw, prob_ow, prob_ol, prob_rl)
    """
    # Ensure inputs are treated as floats/arrays
    pi_i = np.asarray(pi_home, dtype=float)
    pi_j = np.asarray(pi_away, dtype=float)
    
    # Calculate terms explicitly
    # Term 1: gamma * pi_i
    term_rw = gamma * pi_i
    
    # Term 2: nu * (gamma * pi_i)^(2/3) * pi_j^(1/3)
    term_ow = nu * (term_rw)**(2/3) * (pi_j)**(1/3)
    
    # Term 3: nu * (gamma * pi_i)^(1/3) * pi_j^(2/3)
    term_ol = nu * (term_rw)**(1/3) * (pi_j)**(2/3)
    
    # Term 4: pi_j
    term_rl = pi_j
    
    # Denominator
    denominator = term_rw + term_ow + term_ol + term_rl
    
    # Probabilities
    prob_rw = term_rw / denominator
    prob_ow = term_ow / denominator
    prob_ol = term_ol / denominator
    prob_rl = term_rl / denominator
    
    return prob_rw, prob_ow, prob_ol, prob_rl

def calculate_log_likelihood(params, games_df, team_to_idx):
    """
    Calculates the negative log likelihood for the dataset given parameters.
    This is a placeholder for the optimization step.
    
    params: list/array of [pi_1, ..., pi_N, nu, gamma]
    games_df: DataFrame containing game outcomes
    team_to_idx: dict mapping team names to parameter indices
    """
    # This function would be implemented in the next steps using calculate_probabilities
    pass

if __name__ == "__main__":
    # Test the function with example values
    pi_h = 1.2
    pi_a = 0.9
    nu_val = 0.8
    gamma_val = 1.1
    
    probs = calculate_probabilities(pi_h, pi_a, nu_val, gamma_val)
    
    print(f"Testing calculate_probabilities with:")
    print(f"  pi_home={pi_h}, pi_away={pi_a}")
    print(f"  nu={nu_val}, gamma={gamma_val}")
    print("-" * 30)
    print(f"P(RW): {probs[0]:.4f}")
    print(f"P(OW): {probs[1]:.4f}")
    print(f"P(OL): {probs[2]:.4f}")
    print(f"P(RL): {probs[3]:.4f}")
    print("-" * 30)
    print(f"Sum: {sum(probs):.4f}")
    
    # Verify logical consistency:
    # If home is much stronger, P(RW) should be high
    probs_strong_home = calculate_probabilities(5.0, 1.0, nu_val, gamma_val)
    print(f"\nStrong Home Team (5.0 vs 1.0): P(RW) = {probs_strong_home[0]:.4f}")

    # Load real data to confirm accessibility
    data_path = r"d:\A\Warton\Data\whl_2025_games.csv"
    if os.path.exists(data_path):
        df_games = pd.read_csv(data_path)
        print(f"\nSuccessfully accessed games data. Shape: {df_games.shape}")
