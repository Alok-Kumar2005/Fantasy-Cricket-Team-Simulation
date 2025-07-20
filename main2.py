import pandas as pd
import numpy as np
import random
import itertools
import os
import time
from collections import defaultdict

def load_player_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded player data with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None

def generate_valid_teams(player_df):
    """Generate all valid teams that meet role constraints"""
    player_roles = {}
    for _, row in player_df.iterrows():
        player_roles[row['player_code']] = row['role']
    
    all_players = list(player_df['player_code'])
    required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
    
    valid_teams = []
    ### getting all teams
    all_combinations = list(itertools.combinations(all_players, 11))
    
    for comb in all_combinations:
        roles_present = set()
        for player in comb:
            roles_present.add(player_roles[player])
        ### checking for each role
        if required_roles.issubset(roles_present):
            valid_teams.append(tuple(sorted(comb)))
    
    print(f"Generated {len(valid_teams)} valid teams")
    return valid_teams

def evaluate_accuracy(current_counts, target_counts, total_teams):
    """Evaluate how many players are within 5% error"""
    within = 0
    for player in current_counts:
        target_perc = target_counts[player] / total_teams
        actual_perc = current_counts[player] / total_teams
        if target_perc == 0:
            error = 0 if actual_perc == 0 else float('inf')
        else:
            error = (actual_perc - target_perc) / target_perc
        if abs(error) <= 0.05:
            within += 1
    return within

def select_teams_greedy(valid_teams, target_counts, num_teams=20000):
    """Greedy team selection to minimize squared error"""
    all_players = list(target_counts.keys())
    current_counts = {player: 0 for player in all_players}
    selected_teams = []
    selected_set = set()
    
    target_sq = sum(target_counts[p]**2 for p in all_players)
    random.shuffle(valid_teams)
    player_teams = defaultdict(list)
    for team in valid_teams:
        for player in team:
            player_teams[player].append(team)
    
    ## tine start
    start_time = time.time()
    
    for i in range(num_teams):
        if i % 1000 == 0:
            ### current loss
            loss = 0
            for p in all_players:
                diff = current_counts[p] - target_counts[p]
                loss += diff * diff
            within = evaluate_accuracy(current_counts, target_counts, i+1)
            print(f"Team {i+1}: Loss = {loss}, Players within 5% = {within}")
        
        best_team = None
        best_loss = float('inf')
        candidates = []
        
        ### under sampled players
        under_sampled = [p for p in all_players if current_counts[p] < target_counts[p]]
        
        if under_sampled:
            ### if present then focous on them
            focus_player = random.choice(under_sampled)
            candidates = random.sample(player_teams[focus_player], min(1000, len(player_teams[focus_player])))
        else:
            ## if not present the randomly selecting 
            candidates = random.sample(valid_teams, min(1000, len(valid_teams)))
        
        # Evaluate candidates
        for team in candidates:
            if team in selected_set:
                continue
                
            temp_counts = current_counts.copy()
            for p in team:
                temp_counts[p] += 1
                
            ### new loss
            new_loss = 0
            for p in all_players:
                diff = temp_counts[p] - target_counts[p]
                new_loss += diff * diff
                
            if new_loss < best_loss:
                best_loss = new_loss
                best_team = team
                
        # Select the best team
        if best_team is None:
            # Fallback to random selection
            while True:
                team = random.choice(valid_teams)
                if team not in selected_set:
                    best_team = team
                    break
        
        # Update counts and add team
        for p in best_team:
            current_counts[p] += 1
        selected_teams.append(best_team)
        selected_set.add(best_team)
    
    print(f"Team selection completed in {time.time()-start_time:.2f} seconds")
    return selected_teams, current_counts

def build_team_df(selected_teams, player_df):
    """Build final team DataFrame"""
    player_map = {}
    for _, row in player_df.iterrows():
        player_map[row['player_code']] = row
    
    rows = []
    for team_id, team in enumerate(selected_teams, 1):
        for player_code in team:
            player_info = player_map[player_code]
            rows.append({
                'match_code': player_info['match_code'],
                'player_code': player_code,
                'player_name': player_info['player_name'],
                'role': player_info['role'],
                'team': player_info['team'],
                'perc_selection': player_info['perc_selection'],
                'team_id': team_id
            })
    
    return pd.DataFrame(rows)

def evaluate_team_accuracy(team_df):
    """Evaluation function provided in the problem statement"""
    print("🔍 Evaluating Fantasy Team Accuracy...\n")
    print(f"📐 team_df shape: {team_df.shape}")
    total_teams = team_df['team_id'].nunique()
    total_players = team_df['player_code'].nunique()
    print(f"👥 Total unique teams: {total_teams}")
    print(f"🎯 Total unique players: {total_players}")

    role_per_team = team_df.groupby('team_id')['role'].nunique()
    missing_role_teams = role_per_team[role_per_team < 4].count()
    print(f"⚠️ Teams missing at least one role: {missing_role_teams} / {total_teams}\n")

    player_ref = team_df.drop_duplicates(subset='player_code')[
        ['match_code', 'player_code', 'player_name', 'role', 'team', 'perc_selection']
    ].copy()

    team_counts = team_df.groupby('player_code')['team_id'].nunique().reset_index(name='actual_team_count')
    merged = pd.merge(player_ref, team_counts, on='player_code', how='left')
    merged['actual_team_count'] = merged['actual_team_count'].fillna(0).astype(int)

    merged['expected_team_count'] = (merged['perc_selection'] * total_teams).round(0).astype(int)
    merged['actual_perc_selection'] = merged['actual_team_count'] / total_teams

    merged['perc_error'] = (
        (merged['actual_perc_selection'] - merged['perc_selection']) / merged['perc_selection']
    ).round(4)

    merged['perc_selection'] = (merged['perc_selection'] * 100).round(2)
    merged['actual_perc_selection'] = (merged['actual_perc_selection'] * 100).round(2)
    merged['perc_error'] = (merged['perc_error'] * 100).round(2)

    accuracy_df = merged[[
        'match_code', 'player_code', 'player_name', 'role', 'team',
        'perc_selection', 'expected_team_count', 'actual_team_count',
        'actual_perc_selection', 'perc_error'
    ]].sort_values('player_code')

    within_5 = accuracy_df[accuracy_df['perc_error'].abs() <= 5]
    outside_5 = accuracy_df[accuracy_df['perc_error'].abs() > 5]

    print("📊 Accuracy KPIs:")
    print(f"✅ Players within ±5% relative error: {within_5.shape[0]} / {accuracy_df.shape[0]}")
    print(f"❌ Players outside ±5% error: {outside_5.shape[0]}")
    print(f"📉 Minimum error: {accuracy_df['perc_error'].min():.2f}%")
    print(f"📈 Maximum error: {accuracy_df['perc_error'].max():.2f}%\n")

    if not outside_5.empty:
        print("🚨 Players with >5% relative error:\n")
        print(outside_5[['player_code', 'player_name', 'perc_selection',
                         'actual_perc_selection', 'perc_error']].to_string(index=False))

    accuracy_df.to_csv("accuracy_summary2.csv", index=False)
    print("\n✅ Accuracy summary saved as 'accuracy_summary2.csv'")

    return accuracy_df

def main():
    os.makedirs("output", exist_ok=True)
    player_df = load_player_data("data/player_data_sample.csv")
    if player_df is None:
        return
    
    random.seed(42)
    np.random.seed(42)
    
    valid_teams = generate_valid_teams(player_df)
    
    target_counts = {}
    for _, row in player_df.iterrows():
        target_counts[row['player_code']] = round(row['perc_selection'] * 20000)
    
    print("Selecting teams with greedy approach")
    selected_teams, final_counts = select_teams_greedy(valid_teams, target_counts)
    
    print("Building final team DataFrame...")
    team_df = build_team_df(selected_teams, player_df)
    team_df.to_csv("output/team_df.csv", index=False)
    
    accuracy_df = evaluate_team_accuracy(team_df)

if __name__ == "__main__":
    main()