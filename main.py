import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os


def load_player_data(filepath):
    """
    Load csv data of players
    """
    try:
        df = pd.read_csv(filepath)
        print(f"loded df with shape {df.shape} ")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    
def create_weighted_player_pool(player_df):
    """
    Creating weighted pool for each category of players
    """
    role_pools = {}
    
    for role in player_df['role'].unique():
        role_players = player_df[player_df['role'] == role].copy()
        
        ###  Creating weighted selection as per perc_selection
        players_list = []
        weights_list = []
        
        for _, player in role_players.iterrows():
            players_list.append(player)
            weights_list.append(player['perc_selection'])
        
        role_pools[role] = {
            'players': players_list,
            'weights': weights_list
        }
    
    return role_pools


def generate_single_team(role_pools, team_id, player_df):
    """
    getting single teams with all roles
    """
    selected_players = []
    selected_codes = set()  # to store unique

    required_roles = ['WK', 'Batsman', 'Bowler', 'Allrounder']
    
    # to maintain the requirement, at least 1 player from required roles
    # selecting one from each categories
    for role in required_roles:
        if role not in role_pools:
            continue
            
        pool = role_pools[role]
        
        # filtering already selected players
        available_indices = []
        available_weights = []
        
        for i, player in enumerate(pool['players']):
            if player['player_code'] not in selected_codes:
                available_indices.append(i)
                available_weights.append(pool['weights'][i])
        
        if available_indices and available_weights:
            # Normalize weights
            weights_array = np.array(available_weights)
            weights_array = weights_array / weights_array.sum()
            
            # Select random index
            selected_idx = np.random.choice(available_indices, p=weights_array)
            selected_player = pool['players'][selected_idx]
            
            selected_players.append(selected_player)
            selected_codes.add(selected_player['player_code'])
    
    # now selecting rest of 7 players from remaining
    while len(selected_players) < 11:
        all_available_indices = []
        all_available_weights = []
        all_players = []
        
        for role in role_pools:
            pool = role_pools[role]
            for i, player in enumerate(pool['players']):
                if player['player_code'] not in selected_codes:
                    all_available_indices.append(len(all_players))
                    all_available_weights.append(pool['weights'][i])
                    all_players.append(player)
        
        if not all_players:
            break
            
        # Normalize weights
        weights_array = np.array(all_available_weights)
        weights_array = weights_array / weights_array.sum()
        
        # Select random player
        selected_idx = np.random.choice(all_available_indices, p=weights_array)
        selected_player = all_players[selected_idx]
        
        selected_players.append(selected_player)
        selected_codes.add(selected_player['player_code'])
    
    team_data = []
    for player in selected_players:
        team_data.append({
            'match_code': player['match_code'],
            'player_code': player['player_code'],
            'player_name': player['player_name'],
            'role': player['role'],
            'team': player['team'],
            'perc_selection': player['perc_selection'],
            'team_id': team_id
        })
    
    return pd.DataFrame(team_data)



def adjust_selection_probabilities(role_pools, target_counts, current_counts, total_teams_remaining):
    """
    Dynamically adjust probabilities to meet target frequencies
    """
    adjusted_pools = {}
    
    for role in role_pools:
        adjusted_pools[role] = {
            'players': role_pools[role]['players'].copy(),
            'weights': []
        }
        
        for i, player in enumerate(role_pools[role]['players']):
            player_code = player['player_code']
            
            target = target_counts[player_code]
            current = current_counts[player_code]
            remaining_need = max(0, target - current)
            
            ### Adjusting weight based on remaining need
            if total_teams_remaining > 0:
                adjusted_weight = remaining_need / total_teams_remaining
                ### minimum weight to avoid zero probabilities
                adjusted_weight = max(adjusted_weight, 0.001)
            else:
                adjusted_weight = role_pools[role]['weights'][i]
            
            adjusted_pools[role]['weights'].append(adjusted_weight)
    
    return adjusted_pools


def generate_teams(player_df, num_teams=20000):
    """
    Generating fantasy teams 
    """
    print("Generating Teams")
    ### claculating target count for each player
    target_counts = {}
    for _, player in player_df.iterrows():
        target_counts[player['player_code']] = int(player['perc_selection'] * num_teams)
    
    current_counts = defaultdict(int)
    all_teams = []
    unique_teams = set()
    
    ## Creating initial role pools
    role_pools = create_weighted_player_pool(player_df)
    
    team_id = 1
    attempts = 0
    max_attempts = num_teams * 3  ## these will prevent to infine loop 
    
    progress_interval = num_teams // 20
    
    while len(all_teams) < num_teams and attempts < max_attempts:
        attempts += 1
        
        ### Adjust probability after generating 1000 teams
        if len(all_teams) > 0 and len(all_teams) % 1000 == 0:
            remaining_teams = num_teams - len(all_teams)
            role_pools = adjust_selection_probabilities(
                create_weighted_player_pool(player_df), 
                target_counts, 
                current_counts, 
                remaining_teams
            )
        
        ### Generating  a unique tema using generate_single_team fnction 
        team_df = generate_single_team(role_pools, team_id, player_df)
        
        if len(team_df) == 11:
            ### checking for unique team
            team_signature = tuple(sorted(team_df['player_code'].tolist()))
            
            if team_signature not in unique_teams:
                unique_teams.add(team_signature)
                all_teams.append(team_df)
                
                for player_code in team_df['player_code']:
                    current_counts[player_code] += 1
                
                team_id += 1
                
                if len(all_teams) % progress_interval == 0:
                    print("", end="", flush=True)
    
    print(f"Generated {len(all_teams)} unique teams in {attempts} attempts")
    
    ### combining all teams
    final_df = pd.concat(all_teams, ignore_index=True)
    return final_df


### Evaluaing metrics as per given
def evaluate_team_accuracy(team_df):
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

    np.random.seed(42)
    random.seed(42)
    team_df = generate_teams(player_df, num_teams=20000)
    team_df.to_csv("output/team_df.csv", index=False)
    print(f"shape of combeined teams: {team_df.shape}\n")
    
    accuracy_df = evaluate_team_accuracy(team_df)

if __name__ == "__main__":
    main()
