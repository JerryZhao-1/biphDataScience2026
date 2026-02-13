import pandas as pd
import os

def display_power_rankings():
    input_file = r"d:\A\Warton\Data\whl_team_strengths.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run optimize_bradley_terry.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(input_file)
    
    # 2. Rename Columns
    if 'Strength' in df.columns:
        df = df.rename(columns={'Strength': 'Strength_Pi'})
    elif 'Strength_Pi' not in df.columns:
        print("Warning: 'Strength' column not found, using first available float column.")
        # Fallback logic if needed, but assuming standard format
        
    # 3. Sort Descending
    df = df.sort_values('Strength_Pi', ascending=False).reset_index(drop=True)
    
    # 4. Add Rank Column
    df['Rank'] = df.index + 1
    
    # Reorder columns
    final_df = df[['Rank', 'Team', 'Strength_Pi']]
    
    # 5. Display Top 10
    print("\n--- Top 10 WHL Teams (Bradley-Terry Power Rankings) ---")
    print(final_df.head(10).to_string(index=False))
    
    # Save for reference
    output_file = r"d:\A\Warton\Data\whl_power_rankings.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nFull rankings saved to {output_file}")

if __name__ == "__main__":
    display_power_rankings()
