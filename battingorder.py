'''
    File: battingorder.py
    Author: Drew Scott, 2021
    Description:
        Simulates every possible batting order for a given lineup and displays summary statistics of
        the average runs (over a number of games) of the orders, and specific information about the
        top and bottom 5 orders. The user may choose how many games each order is simulated for,
        which players are in the lineup, or what each player's PA outcomes are in a single game
        (see sample files).
    Major TODOs:
        1) Determine the accurate likelihoods of sacrifices and double plays for different out types
        2) Enable some sort of pinch hitting scheme (right now, each player hits the entire game)
        3) Enable some sort of pitcher dependent hitting results (even as simple as right-y/left-y;
            right now hitting stats are full season aggregates)
        4) Include edge cases like stealing, wild pitches, etc.
        5) More detailed data (i.e. did a double go to right or left field?)
        6) Player specific data (i.e. how often does a player try to steal?)
    Usage:
        For a random lineup:
            python3 battingorder.py -n <simulations_per_order>
        For a prespecified lineup of 9 players, format it like giants.txt (naming as in stats.csv)
            python3 battingorder.py -lf <lineup filename> -n <simulations_per_order>
        For a prespecified lineup of 9 players and at-bat outcomes (format like giants_outcomes.txt)
            python3 battingorder.py -lf <lineup filename> -of <outcomes filename>
'''

import timeit
import sys
from typing import Optional, Tuple, List

from simulation import Simulation
from lineup import Lineup
from player import Player

def get_lineup(lineup_filename: Optional[str]) -> Lineup:
    # TODO:
    '''
        Sets the lineup for the simulation
    '''
    lineup = Lineup()

    if lineup_filename is not None:
        lineup.set_players(lineup_filename)
    else:
        lineup.generate_random_lineup()

    return lineup

def parse_arguments(args: List[str]) -> Tuple[Optional[str], Optional[str], int]:
    '''
        Parses the commandline arguments
        Possible options:
            -f (user defined lineup filename, see giants.txt)
            -n (number of simulations per batting order)
    '''

    lineup_filename = None
    outcome_filename = None
    sims_per_order = 1

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

def display_time(start: float, end: float) -> None:
    '''
        Prints hh:mm:ss format of time between start and end
    '''
    seconds = int(end - start)
    hours = seconds // 3600
    minutes = (seconds - (hours * 3600)) // 60
    seconds = seconds - (hours * 3600) - (minutes * 60)
    print(f'Total time: {hours}:{minutes:02d}:{seconds:02d}')

def main():
    '''
        TODO
    '''

    start = timeit.default_timer()

    Player.set_metadata('stats.csv')

    lineup_filename, outcome_filename, sims_per_order = parse_arguments(sys.argv[1:])

    lineup = get_lineup(lineup_filename)
    lineup.set_pa_outcomes(outcome_filename, sims_per_order)

    print("Players:")
    for player in lineup.players:
        print(player)
    print()

    sim = Simulation(sims_per_order, lineup)
    sim.run_sim()

    end = timeit.default_timer()

    display_time(start, end)

if __name__ == '__main__':
    main()
