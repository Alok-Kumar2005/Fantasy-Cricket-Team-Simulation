import pandas as pd
import numpy as np
import random
import itertools
import os
import time
from collections import defaultdict
from tqdm import tqdm

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
    # Create player role mapping
    player_roles = {}
    for _, row in player_df.iterrows():
        player_roles[row['player_code']] = row['role']
    
    all_players = list(player_df['player_code'])
    required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
    
    valid_teams = []
    # Generate all combinations of 11 players
    all_combinations = list(itertools.combinations(all_players, 11))
    
    for comb in all_combinations:
        roles_present = set()
        for player in comb:
            roles_present.add(player_roles[player])
        # Check if all required roles are present
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
    """Enhanced greedy team selection to minimize squared error"""
    # Initialize counts
    all_players = list(target_counts.keys())
    current_counts = {player: 0 for player in all_players}
    selected_teams = []
    selected_set = set()
    
    # Precompute target squared for constant term in loss
    target_sq = sum(target_counts[p]**2 for p in all_players)
    
    # Shuffle valid teams to introduce randomness
    random.shuffle(valid_teams)
    
    # Precompute player-team membership
    player_teams = defaultdict(list)
    for team in valid_teams:
        for player in team:
            player_teams[player].append(team)
    
    # Initialize deficit tracking
    deficit_history = []
    min_deficit = float('inf')
    
    # Start timer
    start_time = time.time()
    pbar = tqdm(total=num_teams, desc="Generating teams")
    
    for i in range(num_teams):
        # Calculate current deficit
        total_deficit = 0
        for player in all_players:
            deficit = max(0, target_counts[player] - current_counts[player])
            total_deficit += deficit
        deficit_history.append(total_deficit)
        
        # Track minimum deficit
        if total_deficit < min_deficit:
            min_deficit = total_deficit
        
        # Check progress every 1000 teams
        if i % 1000 == 0:
            # Calculate current loss
            loss = 0
            for p in all_players:
                diff = current_counts[p] - target_counts[p]
                loss += diff * diff
            within = evaluate_accuracy(current_counts, target_counts, i+1)
            
            # Early stopping if accuracy target is met
            if within >= 20:
                print(f"\nAccuracy target met at team {i+1}")
                break
            
            # Dynamic candidate pool sizing based on deficit
            candidate_pool_size = min(5000, max(1000, int(5000 * (total_deficit / min_deficit))))
            pbar.set_description(f"Teams: {i+1}, Loss: {loss}, Within±5%: {within}, Candidates: {candidate_pool_size}")
        
        # Find under-sampled players
        under_sampled = [p for p in all_players if current_counts[p] < target_counts[p]]
        
        # If we have under-sampled players, focus on the most deficient ones
        if under_sampled:
            # Sort by deficit (largest deficit first)
            under_sampled.sort(key=lambda p: target_counts[p] - current_counts[p], reverse=True)
            
            # Focus on the top 5 most deficient players
            focus_players = under_sampled[:min(5, len(under_sampled))]
            
            # Collect candidate teams containing these focus players
            candidates = []
            for player in focus_players:
                # Get teams containing this player
                teams_for_player = player_teams[player]
                
                # Filter out already selected teams
                available_teams = [t for t in teams_for_player if t not in selected_set]
                
                # Sample a subset if too many
                if len(available_teams) > 1000:
                    candidates.extend(random.sample(available_teams, 1000))
                else:
                    candidates.extend(available_teams)
            
            # Remove duplicates
            candidates = list(set(candidates))
        else:
            # If no under-sampled, use random candidates
            candidates = [t for t in valid_teams if t not in selected_set]
        
        # If we have too many candidates, sample a subset
        if len(candidates) > 5000:
            candidates = random.sample(candidates, 5000)
        
        best_team = None
        best_score = float('-inf')
        
        # Evaluate candidates based on deficit reduction potential
        for team in candidates:
            if team in selected_set:
                continue
                
            # Calculate potential new counts
            temp_counts = current_counts.copy()
            for p in team:
                temp_counts[p] += 1
                
            # Calculate the improvement score
            score = 0
            for p in all_players:
                current_deficit = max(0, target_counts[p] - current_counts[p])
                new_deficit = max(0, target_counts[p] - temp_counts[p])
                improvement = current_deficit - new_deficit
                
                # Weight improvement by how under-sampled the player is
                if current_deficit > 0:
                    weight = 1 + (current_deficit / target_counts[p])
                    score += improvement * weight
            
            if score > best_score:
                best_score = score
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
        pbar.update(1)
    
    pbar.close()
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

    accuracy_df.to_csv("accuracy_summary.csv", index=False)
    print("\n✅ Accuracy summary saved as 'accuracy_summary.csv'")

    return accuracy_df

def main():
    os.makedirs("output", exist_ok=True)
    player_df = load_player_data("data/player_data_sample.csv")
    if player_df is None:
        return
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Generate all valid teams
    print("Generating all valid teams...")
    start_time = time.time()
    valid_teams = generate_valid_teams(player_df)
    print(f"Generated valid teams in {time.time()-start_time:.2f} seconds")
    
    # Step 2: Set target counts
    target_counts = {}
    for _, row in player_df.iterrows():
        target_counts[row['player_code']] = round(row['perc_selection'] * 20000)
    
    # Step 3: Select teams using enhanced greedy algorithm
    print("Selecting teams with enhanced greedy optimization...")
    selected_teams, final_counts = select_teams_greedy(valid_teams, target_counts)
    
    # Step 4: Build final DataFrame
    print("Building final team DataFrame...")
    team_df = build_team_df(selected_teams, player_df)
    team_df.to_csv("output/team_df.csv", index=False)
    print(f"Saved team data with shape {team_df.shape}")
    
    # Step 5: Evaluate accuracy
    accuracy_df = evaluate_team_accuracy(team_df)

if __name__ == "__main__":
    main()