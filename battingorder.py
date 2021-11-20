'''
    File: battingorder.py
    Author: Drew Scott
    Description: Selects a lineup of 9 players from stats.csv and simulates all possible batting orders, determing which results in
        the most average runs per 9 innings. Prints out summary statistics for the simulation.
    Usage: If you want to supply a determined line up of 9 players, format it like giants.txt (ensure naming corresponds to stats.csv)
        and run: python3 battingorder.py <lineup file>
        If you want a random lineup: python3 battingorder.py
'''

import random
from typing import List
import itertools
import sys
import math

def get_player_ratio(stats: dict) -> dict:
    ratios = {}

    total = stats['b_total_pa']

    total_r = 0
    count = 0
    ignore_fields = ['b_total_pa', 'first_name', 'last_name', 'player_id', 'year', 'slg_percent', 'on_base_percent', 'on_base_plus_slg']
    for key, val in stats.items():
        if key in ignore_fields:
            ratios[key] = val
            continue
        
        r = val / total
        ratios[key] = r

        count += val
        total_r += r 

#    print(total_r, count - total)

    return ratios

def get_stats(stats: str, col_names: str) -> dict:
    d = {}

    stats = stats[: -1].split(',')
    stats = [int(s) if s.isnumeric() else s for s in stats]

    col_names = col_names[:-1].split(',')

    for col, stat in zip(col_names, stats):
        d[col] = stat

    return d 

def get_nine(total_count : int) -> List[int]:
    nine = []

    while len(nine) < 9:
        r = random.randint(0, total_count - 1)

        if r not in nine:
            nine.append(r)

    return nine    

def new_batter(cur_batter : int) -> int:
    if cur_batter == 8:
        return 0
    
    return cur_batter + 1

def get_PA_thresholds(player_stat : dict, outcomes : List[str]) -> List[float]:
    thresholds = []
    
    thresholds.append(player_stat[outcomes[0]])
    for i, outcome in enumerate(outcomes[1:], start=1):
        thresholds.append(thresholds[i-1] + player_stat[outcome])

    return thresholds

def get_PA_outcome(threshold : List[float], outcomes : List[str]) -> str:
    rand_outcome = random.random()
    for out_threshold, outcome in zip(threshold, outcomes):
        if rand_outcome < out_threshold:
            return outcome

    return outcomes[-1]    

def sim_inning(player_ratios : dict, player_thresholds : List[float], outcomes : List[str], leadoff : int) -> int:
    runs = 0
    cur_batter = leadoff
    runners = [False, False, False]
    outs = 0

    while outs < 3:
        cur_stats = player_ratios[cur_batter]
        cur_threshold = player_thresholds[cur_batter]

        outcome = get_PA_outcome(cur_threshold, outcomes)
        if outcome[:5] == 'b_out' or outcome == 'b_strikeout':
            # out, need to account for potential sacrficies on fly and ground outs
            outs += 1
            if outs == 3:
                # end of inning, no sacs possible
                break
           
            if outcome == 'b_out_fly':
                third_scores = 0.2
                second_advances = 0.1
                r = random.random()
                if r < third_scores and runners[2] == True:
                    runs += 1
                    runners = [runners[0], runners[1], False]
                
                if r < second_advances and runners[1] == True:
                    runners = [runners[0], False, True] 

            elif outcome == 'b_out_ground':
                sac_rate = 0.001

                if random.random() < sac_rate:
                    runs += int(runners[2])
                    runners = [False, runners[0], runners[1]]

 
        elif outcome == 'b_walk' or outcome == 'b_catcher_interf' or outcome == 'b_hit_by_pitch':
            # right now treating all these the same, just cascade runner advancements along starting with the batter
            if runners[0] == True:
                if runners[1] == True:
                    if runners[2] == True:
                        # runners on first, second, and third
                        runs += 1
                        runners = [True, True, True]

                    else:
                        # runners on first and second
                        runners = [True, True, True]
                else:
                    # runner on first
                    runners = [True, True, runners[2]]
            else:
                # no runner on first
                runners = [True, runners[1], runners[2]]
 
        elif outcome == 'b_single':
            # on a single, runners on second and third score, and runner on first goes to second
            runs += sum(runners[1:])
            runners = [True, runners[0], False]

        elif outcome == 'b_double':
            # on a double, a runner on first advances to third, batter to second, and others advance home
            runs += sum(runners[1:])
            runners = [False, True, runners[0]]

        elif outcome == 'b_triple':
            # all runners score, batter to third
            runs += sum(runners)
            runners = [False, False, True] 

        elif outcome == 'b_home_run':
            # all runners and batter score
            runs += sum(runners) + 1
            runners = [False, False, False]

        cur_batter = new_batter(cur_batter)

    return runs, new_batter(cur_batter)

def run_sim(player_ratios : dict, per_order : int = 1) -> None:
    orders = list(itertools.permutations([i for i in range(9)]))
    
    total_runs = 0
    worst_runs = math.inf
    best_runs = 0
    best_order = None
    outcomes = ['b_single', 'b_double', 'b_triple', 'b_home_run', 'b_strikeout', 'b_walk', 'b_catcher_interf', 'b_hit_by_pitch', 'b_out_fly', 'b_out_ground', 'b_out_line_drive', 'b_out_popup']    
    player_thresholds = [get_PA_thresholds(ratio, outcomes) for ratio in player_ratios]

    for i, order in enumerate(orders):
        print(f'Batting order num: {i}, {i/len(orders)*100:.2f}% complete       ', end='\r')
        tot_runs_order = 0
        for _ in range(per_order):
            leadoff = 0
            for inning in range(9):
                runs, leadoff = sim_inning(player_ratios, player_thresholds, outcomes, leadoff)

                tot_runs_order += runs
    
        avg_runs_order = tot_runs_order / per_order
        if avg_runs_order > best_runs:
            best_runs = avg_runs_order
            best_order = order

        total_runs += avg_runs_order

        if avg_runs_order < worst_runs:
            worst_runs = avg_runs_order

    print('\n')
    print(f'Total games simulated: {len(orders) * per_order:,}')
    print(f'Games simulated per order: {per_order:,}')
    print(f'Total orders simulated: {len(orders):,}')
    print()
    print(f'Max runs for sim: {best_runs}')
    print(f'Avg. runs for sim: {total_runs/len(orders):.2f}')
    print(f'Min runs for sim: {worst_runs}')
    print()
    print(f'Best batting order: {best_order}')
    for i, ind in enumerate(best_order):
        print(f'{str(i+1)}) {player_ratios[ind]}')

if __name__ == '__main__':
    players = []

    # remove all spaces, weird baseball savant thing
    with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
        new_content = ''
        for line in stats_csv:
            updated_line = line.replace(' ', '')
            new_content += updated_line

    with open('stats.csv', 'w', encoding='utf-8-sig') as stats_csv:
        stats_csv.write(new_content)

    if len(sys.argv) == 2:
        # get the players specified in the input file
        lineup = []
        with open(sys.argv[1], 'r') as f_players:
            for line in f_players:
                first, last = line.split()
                lineup.append(f'{last},{first}')

        with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
            col_names = stats_csv.readline().strip()
            
            for line in stats_csv:
                first_comma = line.index(',')
                second_comma = line[first_comma + 1: ].index(',')
                name = line[ : second_comma + first_comma + 1]

                if name in lineup:
                    stats = get_stats(line.strip(), col_names)

                    player_ratio = get_player_ratio(stats)
                    players.append(player_ratio) 
        
    else:
        # get all of the players
        ratios = []
        with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
            col_names = stats_csv.readline().strip()
            
            for line in stats_csv:
                stats = get_stats(line.strip(), col_names)

                player_ratio = get_player_ratio(stats)
                ratios.append(player_ratio) 

        # select 9 random players to run the sim
        player_indexes = get_nine(len(ratios))
        print('Players:')
        for ind in player_indexes:
            players.append(ratios[ind])

    print("Players: ")
    for player in players:
        print(f'{player["first_name"]} {player["last_name"]}: {player["on_base_percent"]} OBP, {player["slg_percent"]} SLG, {player["on_base_plus_slg"]} OPS')
    print()
    
    assert len(players) == 9

    run_sim(players, per_order=2)
