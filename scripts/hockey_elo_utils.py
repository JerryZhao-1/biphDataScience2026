
import pandas as pd
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# -------------------------------------------------------------------
#  Constants (Default Configuration)
# -------------------------------------------------------------------

DEFAULT_ELO_SCALE = 400
DEFAULT_MOV_LOG_BASE = 1
DEFAULT_MOV_CAP = 1.5

# Points System
PTS_REG_WIN = 3
PTS_OT_WIN = 2
PTS_OT_LOSS = 1
PTS_REG_LOSS = 0

# -------------------------------------------------------------------
#  Data Pipeline Functions
# -------------------------------------------------------------------

def load_and_clean(filepath):
    """
    Load raw shift-level CSV and remove duplicates.
    
    IMPORTANT: Empty-net records are KEPT.
    
    Returns
    -------
    pd.DataFrame - Cleaned shift-level data.
    """
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return pd.DataFrame()

    n_original = len(df_raw)
    
    # 1) Remove duplicate records
    if 'record_id' in df_raw.columns:
        dup_mask = df_raw.duplicated(subset=['record_id'])
    else:
        dup_mask = df_raw.duplicated()
    
    n_dups = int(dup_mask.sum())
    df_raw = df_raw.loc[~dup_mask].copy()
    
    # 2) Tag empty-net rows (for optional separate analysis) but DO NOT remove
    empty_net_cols = [
        'home_goalie', 'away_goalie',
        'home_off_line', 'away_off_line',
        'home_def_pairing', 'away_def_pairing'
    ]
    en_mask = pd.Series(False, index=df_raw.index)
    for col in empty_net_cols:
        if col in df_raw.columns:
            en_mask |= df_raw[col].astype(str).str.contains('empty_net', case=False, na=False)
    
    n_empty_net = int(en_mask.sum())
    df_raw['is_empty_net'] = en_mask.astype(int)
    
    print(f"✅ Loaded {n_original:,} records -> Removed {n_dups} duplicates")
    print(f"   Empty-net rows tagged (NOT removed): {n_empty_net} ({n_empty_net/len(df_raw)*100:.1f}%)")
    print(f"   Clean rows: {len(df_raw):,}")
    
    return df_raw


def aggregate_to_games(df_raw):
    """
    Collapse shift-level rows into one row per game.
    Engineer winner, league points, and differentials.
    
    Returns
    -------
    pd.DataFrame - Game-level DataFrame (`df_games`).
    """
    if df_raw.empty:
        return pd.DataFrame()

    agg_rules = {
        'home_goals': 'sum', 'away_goals': 'sum',
        'home_xg':    'sum', 'away_xg':    'sum',
        'home_shots': 'sum', 'away_shots': 'sum',
        'went_ot': 'max',
        'home_team': 'first', 'away_team': 'first',
        'home_assists': 'sum', 'away_assists': 'sum',
        'home_penalties_committed': 'sum', 'away_penalties_committed': 'sum',
        'home_penalty_minutes': 'sum', 'away_penalty_minutes': 'sum',
    }
    df_games = df_raw.groupby('game_id').agg(agg_rules).reset_index()
    
    # Determine winner
    df_games['winner'] = np.where(
        df_games['home_goals'] > df_games['away_goals'], 'Home', 'Away'
    )
    n_draws = (df_games['home_goals'] == df_games['away_goals']).sum()
    if n_draws > 0:
        print(f"   WARNING: {n_draws} draws found — data may be corrupted!")
    
    # League points
    home_win = df_games['home_goals'] > df_games['away_goals']
    away_win = df_games['away_goals'] > df_games['home_goals']
    is_ot    = df_games['went_ot'] == 1
    
    df_games['home_points'] = PTS_REG_LOSS
    df_games.loc[home_win & ~is_ot, 'home_points'] = PTS_REG_WIN
    df_games.loc[home_win &  is_ot, 'home_points'] = PTS_OT_WIN
    df_games.loc[away_win &  is_ot, 'home_points'] = PTS_OT_LOSS
    
    df_games['away_points'] = PTS_REG_LOSS
    df_games.loc[away_win & ~is_ot, 'away_points'] = PTS_REG_WIN
    df_games.loc[away_win &  is_ot, 'away_points'] = PTS_OT_WIN
    df_games.loc[home_win &  is_ot, 'away_points'] = PTS_OT_LOSS
    
    df_games['goal_diff'] = df_games['home_goals'] - df_games['away_goals']
    df_games['xg_diff']   = df_games['home_xg']    - df_games['away_xg']
    
    print(f"✅ Aggregated into {len(df_games)} unique games (draws: {n_draws}).")
    return df_games


def create_long_format(df_games):
    """
    Melt game-level data into two rows per game (home + away perspective).
    """
    cols_to_keep = ['game_id', 'Team', 'Opponent', 'GF', 'GA',
                    'xGF', 'xGA', 'SF', 'SA', 'is_home', 'Result', 'went_ot',
                    'Assists', 'PIM', 'Opp_PIM']
    
    # Home perspective
    df_home = df_games.rename(columns={
        'home_team': 'Team',   'away_team': 'Opponent',
        'home_goals': 'GF',    'away_goals': 'GA',
        'home_xg': 'xGF',     'away_xg': 'xGA',
        'home_shots': 'SF',   'away_shots': 'SA',
        'home_points': 'Result',
        'home_assists': 'Assists',
        'home_penalty_minutes': 'PIM',
        'away_penalty_minutes': 'Opp_PIM',
    }).assign(is_home=1)[cols_to_keep]
    
    # Away perspective
    df_away = df_games.rename(columns={
        'away_team': 'Team',   'home_team': 'Opponent',
        'away_goals': 'GF',    'home_goals': 'GA',
        'away_xg': 'xGF',     'home_xg': 'xGA',
        'away_shots': 'SF',   'home_shots': 'SA',
        'away_points': 'Result',
        'away_assists': 'Assists',
        'away_penalty_minutes': 'PIM',
        'home_penalty_minutes': 'Opp_PIM',
    }).assign(is_home=0)[cols_to_keep]
    
    df_long = pd.concat([df_home, df_away], ignore_index=True)
    df_long = df_long.sort_values(['game_id', 'is_home'], ascending=[True, False]).reset_index(drop=True)
    
    print(f"Long format: {len(df_games)} games -> {len(df_long)} team-game rows")
    return df_long


def calculate_multipliers(df_long, mov_cap=DEFAULT_MOV_CAP, mov_log_base=DEFAULT_MOV_LOG_BASE):
    """
    Add game-level weighting columns used during Elo updates.
    """
    df = df_long.copy()
    
    # Margin-of-Victory multiplier (CAPPED)
    raw_mov = np.log(np.abs(df['GF'] - df['GA']) + mov_log_base)
    df['mov_multiplier'] = np.minimum(raw_mov, mov_cap)
    
    # xG share (handle zero-xG edge cases -> default 0.5)
    total_xg = df['xGF'] + df['xGA']
    df['xg_share'] = np.where(total_xg > 0, df['xGF'] / total_xg, 0.5)

    # Goal share (similar handling)
    total_goals = df['GF'] + df['GA']
    df['goal_share'] = np.where(total_goals > 0, df['GF'] / total_goals, 0.5)

    # Composite Performance Score (0.4 * Goal_Share + 0.6 * xG_Share)
    # mirroring the notebook logic
    df['perf_score'] = 0.4 * df['goal_share'] + 0.6 * df['xg_share']
    
    # Discipline metric
    df['pim_diff'] = df['Opp_PIM'] - df['PIM']
    
    print(f"Multipliers computed. Stats:\n{df[['mov_multiplier', 'xg_share', 'goal_share', 'perf_score']].describe().round(4)}")
    return df

# -------------------------------------------------------------------
#  Elo Core Functions
# -------------------------------------------------------------------

def elo_expected_score(rating_a, rating_b, hfa=0, elo_scale=DEFAULT_ELO_SCALE):
    """
    Logistic expected score for Team A against Team B.
    E(A) = 1 / (1 + 10^((R_B - (R_A + HFA)) / K))
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - (rating_a + hfa)) / elo_scale))


def margin_of_victory_multiplier(goal_diff, mov_cap=DEFAULT_MOV_CAP, mov_log_base=DEFAULT_MOV_LOG_BASE):
    """
    Logarithmic Margin-of-Victory multiplier.
    """
    return min(np.log(abs(goal_diff) + mov_log_base), mov_cap)


def composite_actual_score(result_points, xg_share, w_result=0.3, w_xg=0.7):
    """
    Blend result with process-based (xG) performance.
    """
    result_normalized = result_points / 3.0
    return w_result * result_normalized + w_xg * xg_share


def update_elo_ratings(
    rating_home,
    rating_away,
    result_pts_home,
    xg_share_home,
    xg_share_away,
    mov_mult,
    k_factor,
    hfa,
    w_result=0.3,
    w_xg=0.7,
    elo_scale=DEFAULT_ELO_SCALE
):
    """
    Full Elo update step for a single game.
    """
    exp_home = elo_expected_score(rating_home, rating_away, hfa=hfa, elo_scale=elo_scale)
    exp_away = 1.0 - exp_home

    pts_map = {3: 0, 2: 1, 1: 2, 0: 3}
    result_pts_away = pts_map.get(result_pts_home, 0)

    s_home = composite_actual_score(result_pts_home, xg_share_home, w_result=w_result, w_xg=w_xg)
    s_away = composite_actual_score(result_pts_away, xg_share_away, w_result=w_result, w_xg=w_xg)

    total = s_home + s_away
    if total == 0:
        s_home, s_away = 0.5, 0.5
    else:
        s_home /= total
        s_away /= total

    new_home = rating_home + k_factor * mov_mult * (s_home - exp_home)
    new_away = rating_away + k_factor * mov_mult * (s_away - exp_away)

    return new_home, new_away


# -------------------------------------------------------------------
#  Simulation & Prediction
# -------------------------------------------------------------------

def prepare_simulation_data(df_train):
    """
    Convert long-format DataFrame into a list of game dictionaries for fast iteration.
    """
    df_home_sim = df_train[df_train['is_home'] == 1].set_index('game_id')
    df_away_sim = df_train[df_train['is_home'] == 0].set_index('game_id')

    # Join to get all info in one row
    # We need: Home Team, Away Team, Result (Points), Home Perf, Away Perf, Mov Mult
    sim_games_df = df_home_sim[['Team', 'Result', 'perf_score', 'mov_mult']].join(
        df_away_sim[['Team', 'perf_score']], lsuffix='_home', rsuffix='_away'
    )

    games_list = []
    for _, row in sim_games_df.iterrows():
        games_list.append({
            'home': row['Team_home'],
            'away': row['Team_away'],
            'home_pts': row['Result'],
            'home_perf': row['perf_score_home'],
            'away_perf': row['perf_score_away'],
            'mov_mult': row['mov_mult']
        })
    
    print(f"Prepared {len(games_list)} games for simulation.")
    return games_list


def run_elo_simulation(df_train, n_sims=1000, k_factor=25, hfa=14.89, w_result=0.3, w_xg=0.7):
    """
    Run the Bagging Elo Simulation.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (long format) with 'perf_score', 'mov_multiplier', etc.
    n_sims : int
        Number of bagging iterations.
    k_factor : float
        K-factor for Elo updates.
    hfa : float
        Home Field Advantage (added to home rating).
        
    Returns
    -------
    pd.DataFrame - Final rankings with Mean Elo and Std Dev.
    """
    games_list = prepare_simulation_data(df_train)
    teams = df_train['Team'].unique()
    
    # Dictionary to store final ratings from each sim: {team: [r1, r2, ...]}
    sim_results = {team: [] for team in teams}

    print(f"Running {n_sims} simulations (K={k_factor}, HFA={hfa})...")
    
    # Use a fixed seed for the OUTER loop logic if desired, but we want randomness.
    # We can seed mostly outside or just let it run.
    # The notebook header said `np.random.seed(42)` before the loop.
    np.random.seed(42)

    for _ in tqdm(range(n_sims)):
        # Init Ratings
        current_ratings = {team: 1500.0 for team in teams}

        # Shuffle games
        daily_games = games_list.copy()
        random.shuffle(daily_games)

        # Play Season
        for g in daily_games:
            home = g['home']
            away = g['away']

            rh = current_ratings[home]
            ra = current_ratings[away]

            # In the notebook, update_ratings calls were:
            # update_ratings(rh, ra, g['home_pts'], g['home_perf'], g['away_perf'], ...)
            # We map home_perf/away_perf to xg_share_home/away arguments of update_elo_ratings
            nrh, nra = update_elo_ratings(
                rating_home=rh,
                rating_away=ra,
                result_pts_home=g['home_pts'],
                xg_share_home=g['home_perf'],
                xg_share_away=g['away_perf'],
                mov_mult=g['mov_mult'],
                k_factor=k_factor,
                hfa=hfa,
                w_result=w_result,
                w_xg=w_xg
            )

            current_ratings[home] = nrh
            current_ratings[away] = nra

        # Store final ratings
        for team, rating in current_ratings.items():
            sim_results[team].append(rating)

    # Aggregation
    final_ratings = []
    for team, ratings in sim_results.items():
        mean_rating = np.mean(ratings)
        std_dev = np.std(ratings)
        # CI logic can be added here or later
        final_ratings.append({'Team': team, 'Elo_Mean': mean_rating, 'Elo_Std': std_dev})

    df_elo = pd.DataFrame(final_ratings).sort_values('Elo_Mean', ascending=False).reset_index(drop=True)
    df_elo['Elo_Rank'] = df_elo.index + 1
    
    print("\nSimulation Complete. Top 10:")
    print(df_elo.head(10))
    
    return df_elo


def predict_matchups(matchup_file, df_elo, hfa=14.89):
    """
    Load matchups and predict outcomes using Elo ratings.
    """
    try:
        # Load matchups
        df_matchups = pd.read_csv(matchup_file)
        
        # Standardize columns (expect Home, Away)
        cols = df_matchups.columns
        # Heuristic: verify if we have 'home_team'/'away_team' or just pick first distinct team-like columns?
        # The notebook had specific logic:
        if 'home_team' in cols and 'away_team' in cols:
            df_matchups = df_matchups[['home_team', 'away_team']].copy()
            df_matchups.columns = ['Home', 'Away']
        elif len(cols) >= 2:
            # Fallback (as per notebook attempt, though it raised error)
            # We'll assume names if they look like Home/Away or just take 0 and 1?
            # Actually, notebook code raised error if cols missing. We should try to be robust.
            pass 
        
        # Merge Ratings
        df_pred = df_matchups.merge(df_elo[['Team', 'Elo_Mean']], left_on='Home', right_on='Team', how='left').rename(columns={'Elo_Mean': 'Elo_Home'})
        df_pred = df_pred.merge(df_elo[['Team', 'Elo_Mean']], left_on='Away', right_on='Team', how='left').rename(columns={'Elo_Mean': 'Elo_Away'})
        
        # Calculate Prob
        # P(HomeWin) = 1 / (1 + 10^((RA - (RH + HFA)) / 400))
        df_pred['Prob_Home'] = 1.0 / (1.0 + 10.0 ** ((df_pred['Elo_Away'] - (df_pred['Elo_Home'] + hfa)) / 400.0))
        
        df_pred['Predicted_Winner'] = np.where(df_pred['Prob_Home'] > 0.5, df_pred['Home'], df_pred['Away'])
        
        # Risk Level
        def get_risk(prob):
            if abs(prob - 0.5) < 0.05: return 'High Risk / Potential OT'
            elif prob > 0.65 or prob < 0.35: return 'Solid Bet'
            return 'Medium Risk'

        df_pred['Risk_Level'] = df_pred['Prob_Home'].apply(get_risk)
        
        return df_pred

    except Exception as e:
        print(f"Error in prediction: {e}")
        return pd.DataFrame()

