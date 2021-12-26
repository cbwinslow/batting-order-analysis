'''
    File: battingorder.py
    Author: Drew Scott
    Description: Uses a lineup of 9 players  and simulates all possible batting orders, determining which results in
        the most average runs per 9 innings. Prints out summary statistics for the simulation.
    Usage:
        For a random lineup:
            python3 battingorder.py -n <simulations_per_order>
        For a prespecified lineup of 9 players, format it like giants.txt (ensure naming corresponds to stats.csv)
            python3 battingorder.py -lf <lineup filename> -n <simulations_per_order>
        For a prespecified lineup of 9 players and at-bat outcomes (format like giants_outcomes.txt)
            python3 battingorder.py -lf <lineup filename> -of <outcomes filename>
'''

from Lineup import Lineup, Player

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
import heapq

def new_batter(cur_batter : int, thru_order:int) -> int:
    '''
        Returns the index of the next batter
    '''
    if cur_batter == 8:
        return 0, thru_order+1

    return cur_batter + 1, thru_order

def sim_inning(generated_outcomes:dict, leadoff : int, thru_order:int, order:List[int]) -> Tuple[int, int]:
    '''
        Simulates an inning of play
        Returns the number of runs scored in the inning and what batter will lead off the next inning
    '''

    runs = 0
    cur_batter_pos = leadoff
    cur_batter = order[cur_batter_pos]
    runners = [False, False, False]
    outs = 0

    while outs < 3:
        # there aren't enough outcomes for this batter, so guarantee an out
        # TODO: actually simulate AB
        if len(generated_outcomes[cur_batter]) <= thru_order:
            outs += 1
            cur_batter_pos, thru_order = new_batter(cur_batter_pos, thru_order)
            cur_batter = order[cur_batter_pos]
            continue 

        outcome = generated_outcomes[cur_batter][thru_order]

        if outcome[:5] == 'b_out' or outcome == 'b_strikeout':
            # out, need to account for potential sacrficies on fly and ground outs
            # and double/triple plays
            outs += 1
            if outs == 3:
                # end of inning, no sacs possible
                break

            # TODO: tweak rates
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
                double_play_rate = 0.3

                if random.random() < sac_rate:
                    runs += int(runners[2])
                    runners = [False, runners[0], runners[1]]
                elif random.random() < double_play_rate:
                    outs += int(runners[0])
                    runners = [False, runners[1], runners[2]]

                    if outs >= 3:
                        break

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
        else:
            raise Exception(f"Invalid outcome name: {outcome}")

        cur_batter_pos, thru_order = new_batter(cur_batter_pos, thru_order)
        cur_batter = order[cur_batter_pos]

    cur_batter_pos, thru_order = new_batter(cur_batter_pos, thru_order)

    return runs, cur_batter_pos, thru_order

def sim_order(order:List[int], per_order:int, lineup:Lineup) -> float:
    '''
        Simulates per_order games for each order
        Parallelized using imap
    '''
    tot_runs_order = 0
    for game_num in range(per_order):
        leadoff = 0
        thru_order = 0
        game_outcomes = lineup.game_outcomes[game_num]
        for inning in range(9):
            runs, leadoff, thru_order = sim_inning(game_outcomes, leadoff, thru_order, order)

            tot_runs_order += runs

    avg_runs_order = tot_runs_order / per_order

    return avg_runs_order

def run_sim(lineup:Lineup, per_order:int = 1) -> None:
    '''
        Simulates per_order games for each possible batting order
    '''

    orders = list(itertools.permutations([i for i in range(9)]))
    # run the simulation with the at bat outcomes
    pool = Pool(10)
    avg_runs_per_order = list(tqdm.tqdm( \
        pool.imap(partial(sim_order, per_order=per_order, lineup=lineup), orders), \
        total=len(orders)))
    pool.close()
    pool.join()

    # aggregate data
    avg_runs_indexes = [(runs, i) for i, runs in enumerate(avg_runs_per_order)]
    best_five = heapq.nlargest(5, avg_runs_indexes)
    worst_five = heapq.nsmallest(5, avg_runs_indexes)
   
    best_runs = max(avg_runs_per_order)
    worst_runs = min(avg_runs_per_order)
    avg_runs = mean(avg_runs_per_order)

    print(f'Total games simulated: {len(orders) * per_order:,}')
    print(f'Games simulated per order: {per_order:,}')
    print(f'Total orders simulated: {len(orders):,}')
    print()
    print(f'Max runs for sim: {best_runs}')
    print(f'Avg. runs for sim: {avg_runs:.2f}')
    print(f'Min runs for sim: {worst_runs}')
    print()

    print(f'Top 5 batting orders:')
    for rank in range(len(best_five)):
        order = orders[best_five[rank][1]] 
        print(f'{str(rank+1)}) Average runs for order: {best_five[rank][0]}')
        for i, ind in enumerate(order):
            print(f'\t{str(i+1)}) {str(lineup.players[ind])}')
        print()

    print(f'Bottom 5 batting orders:')
    for rank in range(len(worst_five)):
        order = orders[worst_five[rank][1]] 
        print(f'{str(rank+1)}) Average runs for order: {worst_five[rank][0]}')
        for i, ind in enumerate(order):
            print(f'\t{str(i+1)}) {str(lineup.players[ind])}')
        print()

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

def get_lineup(lineup_filename:str) -> Lineup:
    lineup = Lineup()

    if lineup_filename is not None:
        # get the players specified in the input file
        player_names = []
        with open(lineup_filename, 'r') as f_players:
            for line in f_players:
                first, last = line.split()
                player_names.append(f'{last},{first}')

        # read the player data from master file
        with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
            col_names = stats_csv.readline().strip()[:-1].split(',')
            for line in stats_csv:
                first_comma = line.index(',')
                second_comma = line[first_comma + 1: ].index(',')
                name = line[ : second_comma + first_comma + 1]

                if name in player_names:
                    lineup.add_player(Player(line))            

    else:
        # get all of the players
        players = []
        with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
            col_names = stats_csv.readline().strip()[:-1].split(',')
            for line in stats_csv:
                players.append(Player(line))

            # select 9 random players to run the sim
            player_indexes = get_nine(len(ratios))
            for index in player_indexes:
                lineup.add_player(players[index])

    # TODO: should this check be inside the Lineup class?
    if len(lineup.players) != 9:
        raise Exception(f'Incorrect number of players: {len(lineup.players)}, {lineup.players}')

    return lineup

def parse_arguments(args):
    '''
        Parses the commandline arguments
        Possible options: -f (user defined lineup filename, see giants.txt) and -n (number of simulations per batting order)
    '''

    lineup_filename = None
    sims_per_order = 1
    outcome_filename = None

    for i in range(len(args) // 2):
        opt_index = i*2
        arg_index = opt_index + 1
        if args[opt_index] == '-lf':
            lineup_filename = args[arg_index]
        elif args[opt_index] == '-n':
            sims_per_order = int(args[arg_index])
        elif args[opt_index] == '-of':
            outcome_filename = args[arg_index]

    if outcome_filename is not None:
        sims_per_order = 1

    return lineup_filename, outcome_filename, sims_per_order

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

    lineup_filename, outcome_filename, sims_per_order = parse_arguments(sys.argv[1:])
    lineup = get_lineup(lineup_filename)
    
    print("Players:")
    for player in lineup.players:
        print(player)
    print()

    lineup.set_pa_outcomes(outcome_filename, sims_per_order)

    # set per_order to be the number of simulations to run per order
    run_sim(lineup, per_order=sims_per_order)

    end = timeit.default_timer()
    seconds = int(end - start)
    hours = seconds // 3600
    minutes = (seconds - (hours * 3600)) // 60
    seconds = seconds - (hours * 3600) - (minutes * 60)
    print(f'Total time: {hours}:{minutes:02d}:{seconds:02d}')
