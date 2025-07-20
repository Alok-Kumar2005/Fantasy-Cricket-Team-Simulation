# Fantasy-Cricket-Team-Simulation

## Environment Setup and download requirements
```
uv init
uv venv
.venv\Scripts\activate
uv add -r requirements.txt
```
## Run file
```
python main.py
```


## Important
- Generate ~20,000 unique fantasy teams
- Print detailed accuracy metrics

- Passes 20,000 teams but not give exactly 20,000 teams becuase of generating some similar teams 
- so to get the 20,000 teams we have to chnage these parameter and these is beacuse 
- Dynamic Adjustment: Adjusts probabilities during generation to meet target frequencies
Uniqueness Check: Prevents duplicate teams

```
📐 team_df shape: (220000, 7)
👥 Total unique teams: 20000
🎯 Total unique players: 22
⚠️ Teams missing at least one role: 0 / 20000

📊 Accuracy KPIs:
✅ Players within ±5% relative error: 5 / 22
❌ Players outside ±5% error: 17
📉 Minimum error: -13.88%
📈 Maximum error: 395.58%

🚨 Players with >5% relative error:

 player_code player_name  perc_selection  actual_perc_selection  perc_error
           1   Player_20           39.57                  55.30       39.75
           2    Player_2           21.31                  52.32      145.52
           3    Player_7           91.91                  81.16      -11.70
           4   Player_12           95.22                  82.00      -13.88
           5   Player_10           21.45                  27.36       27.55
           7   Player_22           89.19                  79.47      -10.89
           8   Player_11           27.09                  32.30       19.23
          10    Player_3            2.83                  14.03      395.58
          13    Player_9           36.18                  39.66        9.62
          13    Player_9           36.18                  39.66        9.62
          14   Player_14           25.36                  30.80       21.47
          15   Player_17           93.60                  81.65      -12.77
          16   Player_19           35.18                  39.08       11.09
          14   Player_14           25.36                  30.80       21.47
          15   Player_17           93.60                  81.65      -12.77
          16   Player_19           35.18                  39.08       11.09
          15   Player_17           93.60                  81.65      -12.77
          16   Player_19           35.18                  39.08       11.09
          16   Player_19           35.18                  39.08       11.09
          18   Player_13           10.68                  17.62       65.03
          19    Player_5           83.48                  77.02       -7.74
          20   Player_16           30.60                  35.51       16.05
          21    Player_6            3.30                  14.16      329.24
          22   Player_15           30.03                  35.11       16.92

✅ Accuracy summary saved as 'accuracy_summary.csv'
```


```
📐 team_df shape: (220000, 7)
👥 Total unique teams: 20000
🎯 Total unique players: 22
⚠️ Teams missing at least one role: 0 / 20000

📊 Accuracy KPIs:
✅ Players within ±5% relative error: 4 / 22
❌ Players outside ±5% error: 18
📉 Minimum error: -10.51%
📈 Maximum error: 401.59%

🚨 Players with >5% relative error:

 player_code player_name  perc_selection  actual_perc_selection  perc_error
           1   Player_20           39.57                  60.30       52.39
           2    Player_2           21.31                  44.54      109.01
           3    Player_7           91.91                  82.74       -9.98
           4   Player_12           95.22                  85.22      -10.51
           5   Player_10           21.45                  28.02       30.63
           7   Player_22           89.19                  80.68       -9.54
           8   Player_11           27.09                  32.41       19.64
           9   Player_21           69.35                  65.53       -5.51
          10    Player_3            2.83                  14.19      401.59
          13    Player_9           36.18                  39.49        9.15
          14   Player_14           25.36                  31.05       22.44
          15   Player_17           93.60                  83.98      -10.27
          16   Player_19           35.18                  38.71       10.05
          18   Player_13           10.68                  19.93       86.66
          19    Player_5           83.48                  76.76       -8.05
          20   Player_16           30.60                  35.30       15.34
          21    Player_6            3.30                  14.52      340.15
          22   Player_15           30.03                  34.84       16.03

✅ Accuracy summary saved as 'accuracy_summary2.csv'
```