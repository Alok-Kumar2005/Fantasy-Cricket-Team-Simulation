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