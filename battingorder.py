'''
    File: battingorder.py
    Author: Drew Scott
    Description: Selects a lineup of 9 players from stats.csv and simulates all possible batting orders, determing which results in
        the most average runs per 9 innings. Prints out summary statistics for the simulation.
    Usage: If you want to supply a determined line up of 9 players, format it like giants.txt (ensure naming corresponds to stats.csv)
        python3 battingorder.py -f <filename> -n <simulations_per_order>
'''

import random
from typing import List, Tuple
import itertools
import sys
import math
import timeit
from multiprocessing import Pool
from functools import partial
from statistics import mean
import tqdm

def get_player_ratio(stats: dict) -> dict:
    '''
        Returns a dict containing only the possible plate appearance outcomes for the input player
    '''
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


    return ratios

def get_stats(stats: str, col_names: str) -> dict:
    '''
        Returns a dict of all a player's stats
    '''
    d = {}

    stats = stats[: -1].split(',')
    stats = [int(s) if s.isnumeric() else s for s in stats]

    col_names = col_names[:-1].split(',')

    for col, stat in zip(col_names, stats):
        d[col] = stat

    return d

def get_nine(total_count : int) -> List[int]:
    '''
        Returns a list of length 9 with unique indexes in the range of total_count
    '''
    nine = []

    while len(nine) < 9:
        r = random.randint(0, total_count - 1)

        if r not in nine:
            nine.append(r)

    return nine

def new_batter(cur_batter : int, thru_order:int) -> int:
    '''
        Returns the index of the next batter
    '''
    if cur_batter == 8:
        return 0, thru_order+1

    return cur_batter + 1, thru_order

def get_PA_thresholds(player_stat : dict, outcomes : List[str]) -> List[float]:
    '''
        Returns the upper threshold (between 0 and 1) for each outcome. Used to decide a plate appearance outcome using a random number
    '''
    thresholds = []

    thresholds.append(player_stat[outcomes[0]])
    for i, outcome in enumerate(outcomes[1:], start=1):
        thresholds.append(thresholds[i-1] + player_stat[outcome])

    return thresholds

def get_PA_outcome(threshold : List[float], outcomes : List[str]) -> str:
    '''
        Returns the name of the plate appearance outcome based on a randomly generated number
    '''
    rand_outcome = random.random()
    for out_threshold, outcome in zip(threshold, outcomes):
        if rand_outcome < out_threshold:
            return outcome

    return outcomes[-1]

def sim_inning(generated_outcomes:dict, leadoff : int, thru_order:int) -> Tuple[int, int]:
    '''
        Simulates an inning of play
        Returns the number of runs scored in the inning and what batter will lead off the next inning
    '''
    runs = 0
    cur_batter = leadoff
    runners = [False, False, False]
    outs = 0

    while outs < 3:
        outcome = generated_outcomes[cur_batter][thru_order]

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

        cur_batter, thru_order = new_batter(cur_batter, thru_order)

    cur_batter, thru_order = new_batter(cur_batter, thru_order)
    return runs, cur_batter, thru_order

def sim_order(orders:List[int], per_order:int, generated_outcomes:List[dict]) -> float:
    '''
        Simulates per_order games for each order
        Parallelized using imap
    '''
    tot_runs_order = 0
    for game_num in range(per_order):
        leadoff = 0
        thru_order = 0
        game_outcomes = generated_outcomes[game_num]
        for inning in range(9):
            runs, leadoff, thru_order = sim_inning(game_outcomes, leadoff, thru_order)

            tot_runs_order += runs

    avg_runs_order = tot_runs_order / per_order

    return avg_runs_order

def run_sim(player_ratios : dict, per_order : int = 1) -> None:
    '''
        Simulates per_order games for each possible batting order
    '''

    orders = list(itertools.permutations([i for i in range(9)]))[:10000]
    outcomes = ['b_single', 'b_double', 'b_triple', 'b_home_run', 'b_strikeout', 'b_walk', 'b_catcher_interf', 'b_hit_by_pitch', 'b_out_fly', 'b_out_ground', 'b_out_line_drive', 'b_out_popup']
    player_thresholds = [get_PA_thresholds(ratio, outcomes) for ratio in player_ratios]

    # pre-generate 10 at bats for each player per_order times
    # this way, the outcomes are held constant across batting orders, so variance should fall
    generated_outcomes = []
    for _ in range(per_order):
        game_outcomes = {}
        for i, player in enumerate(player_ratios):
            game_outcomes[i] = []
            for _ in range(10):
                outcome = get_PA_outcome(player_thresholds[i], outcomes)
                game_outcomes[i].append(outcome)

        generated_outcomes.append(game_outcomes)

    pool = Pool(10)
    avg_runs_per_order = list(tqdm.tqdm( \
        pool.imap(partial(sim_order, per_order=per_order, generated_outcomes=generated_outcomes), orders), \
        total=len(orders)))
    pool.close()
    pool.join()

    best_runs = max(avg_runs_per_order)
    best_order = orders[avg_runs_per_order.index(best_runs)]
    worst_runs = min(avg_runs_per_order)
    worst_order = orders[avg_runs_per_order.index(worst_runs)]
    avg_runs = mean(avg_runs_per_order)

    print(f'Total games simulated: {len(orders) * per_order:,}')
    print(f'Games simulated per order: {per_order:,}')
    print(f'Total orders simulated: {len(orders):,}')
    print()
    print(f'Max runs for sim: {best_runs}')
    print(f'Avg. runs for sim: {avg_runs:.2f}')
    print(f'Min runs for sim: {worst_runs}')
    print()

    print(f'Best batting order: {best_order}')
    for i, ind in enumerate(best_order):
        print(f'{str(i+1)}) {player_summary(player_ratios[ind])}')
    print()

    print(f'Worst batting order: {worst_order}')
    for i, ind in enumerate(worst_order):
        print(f'{str(i+1)}) {player_summary(player_ratios[ind])}')

def player_summary(player : dict) -> str:
    return f'{player["first_name"]} {player["last_name"]}: {player["on_base_percent"]} OBP, {player["slg_percent"]} SLG, {player["on_base_plus_slg"]} OPS'

def parse_arguments(args):
    '''
        Parses the commandline arguments
        Possible options: -f (user defined lineup filename, see giants.txt) and -n (number of simulations per batting order)
    '''

    lineup_filename = None
    sims_per_order = 1

    for i in range(len(args) // 2):
        opt_index = i*2
        arg_index = opt_index + 1
        if args[opt_index] == '-f':
            lineup_filename = args[arg_index]
        elif args[opt_index] == '-n':
            sims_per_order = int(args[arg_index])

    return lineup_filename, sims_per_order

def get_players(lineup_filename):
    players = []

    if lineup_filename is not None:
        # get the players specified in the input file
        lineup = []
        with open(lineup_filename, 'r') as f_players:
            for line in f_players:
                first, last = line.split()
                lineup.append(f'{last},{first}')

        # read the player data from master file
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
            players = [ratios[p] for p in player_indexes]

    return players

if __name__ == '__main__':
    start = timeit.default_timer()
    players = []

    # remove all spaces, weird baseball savant thing
    with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
        new_content = ''
        for line in stats_csv:
            updated_line = line.replace(' ', '')
            new_content += updated_line
    with open('stats.csv', 'w', encoding='utf-8-sig') as stats_csv:
        stats_csv.write(new_content)

    lineup_filename, sims_per_order = parse_arguments(sys.argv[1:])

    players = get_players(lineup_filename)

    print("Players:")
    for player in players:
        print(player_summary(player))
    print()

    assert len(players) == 9

    # set per_order to be the number of simulations to run per order
    run_sim(players, per_order=sims_per_order)

    end = timeit.default_timer()
    print(f'\nTotal time: {end - start:.2f} seconds')
