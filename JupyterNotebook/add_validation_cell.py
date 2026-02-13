
import json
import os

notebook_path = r'd:/A/Warton/JupyterNotebook/ELO.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new cell
new_cell_source = [
    "# ─── Training/Support/Validation Split (80/20) ───\n",
    "print(\"\\n--- Model Validation (80/20 Split) ---\")\n",
    "\n",
    "# 1. Prepare Split\n",
    "# Convert game IDs to integers for sorting (game_1 -> 1, game_10 -> 10)\n",
    "def extract_id(gid):\n",
    "    try:\n",
    "        return int(gid.split('_')[1])\n",
    "    except:\n",
    "        return 0\n",
    "\n",
    "# Get train games (from whl_2025.csv)\n",
    "train_dataset = df_team_stats[df_team_stats['dataset_type'] == 'train']\n",
    "unique_games = train_dataset['game_id'].unique()\n",
    "sorted_games = sorted(unique_games, key=extract_id)\n",
    "\n",
    "split_idx = int(len(sorted_games) * 0.8)\n",
    "train_gids = sorted_games[:split_idx]\n",
    "test_gids = sorted_games[split_idx:]\n",
    "\n",
    "print(f\"Total Games: {len(sorted_games)}\")\n",
    "print(f\"Training Games: {len(train_gids)}\")\n",
    "print(f\"Testing Games: {len(test_gids)}\")\n",
    "\n",
    "# 2. Build Lookup Dictionary\n",
    "# We leverage the fact that df_team_stats has all the pre-calc metrics (mov_mult, xg)\n",
    "# Filter for Home perspective to get one row per game\n",
    "df_home_only = train_dataset[train_dataset['is_home'] == 1].set_index('game_id')\n",
    "\n",
    "games_dict = {}\n",
    "for gid in sorted_games: # Build for all to cover both sets\n",
    "    if gid in df_home_only.index:\n",
    "        row = df_home_only.loc[gid]\n",
    "        games_dict[gid] = {\n",
    "            'home': row['Team'],\n",
    "            'away': row['Opponent'],\n",
    "            'home_pts': row['Result'],\n",
    "            'xg_share': row['xg_share'],\n",
    "            'mov_mult': row['mov_mult']\n",
    "        }\n",
    "\n",
    "# 3. Train on 80%\n",
    "val_ratings = {team: ELO_BASE for team in teams}\n",
    "\n",
    "for gid in train_gids:\n",
    "    g = games_dict[gid]\n",
    "    # Reconstruct inputs for update_ratings\n",
    "    # update_ratings expects: rating_home, rating_away, result_points_home, xg_share_home, xg_share_away, mov_mult\n",
    "    \n",
    "    xg_home = g['xg_share']\n",
    "    xg_away = 1.0 - xg_home # Zero-sum assumption\n",
    "    \n",
    "    nrh, nra = update_ratings(\n",
    "        val_ratings[g['home']], val_ratings[g['away']],\n",
    "        g['home_pts'], xg_home, xg_away,\n",
    "        g['mov_mult'], k=K_FACTOR, hfa=HFA\n",
    "    )\n",
    "    val_ratings[g['home']] = nrh\n",
    "    val_ratings[g['away']] = nra\n",
    "\n",
    "# 4. Predict on 20%\n",
    "correct = 0\n",
    "total = 0\n",
    "\n",
    "for gid in test_gids:\n",
    "    g = games_dict[gid]\n",
    "    \n",
    "    # Calculate Probability\n",
    "    prob_home = calculate_expected_score(val_ratings[g['home']], val_ratings[g['away']], hfa=HFA)\n",
    "    \n",
    "    # Prediction (Threshold 0.5)\n",
    "    if prob_home > 0.5:\n",
    "        pred_winner = g['home']\n",
    "    else:\n",
    "        pred_winner = g['away']\n",
    "        \n",
    "    # Actual Result (Home Pts: 3 or 2 = Win)\n",
    "    # 3 = Reg Win, 2 = OT Win\n",
    "    if g['home_pts'] >= 2:\n",
    "        actual_winner = g['home']\n",
    "    else:\n",
    "        actual_winner = g['away']\n",
    "        \n",
    "    if pred_winner == actual_winner:\n",
    "        correct += 1\n",
    "    total += 1\n",
    "\n",
    "acc = correct / total\n",
    "print(f\"Validation Accuracy: {acc:.2%}\")"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_source
}

# Find insertion point
insert_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = "".join(cell['source'])
        if "# Step 5: Ranking & Prediction" in source_text:
            insert_index = i
            break

if insert_index != -1:
    nb['cells'].insert(insert_index, new_cell)
    print(f"Inserted new cell at index {insert_index}")
else:
    # Append if Step 5 not found (fallback)
    nb['cells'].append(new_cell)
    print("Step 5 header not found, appended cell to the end.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
