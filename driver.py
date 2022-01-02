'''
    File: driver.py
    Author: Drew Scott, 2021
    Usage:
        For a random lineup:
            python3 driver.py -n <simulations_per_order>
        For a prespecified lineup of 9 players (like giants.txt, naming as in stats.csv)
            python3 driver.py -lf <lineup filename> -n <simulations_per_order>
        For a prespecified lineup of 9 players and their PA outcomes (like giants_outcomes.txt)
            python3 driver.py -lf <lineup filename> -of <outcomes filename>
'''

import timeit
import sys
from typing import Optional, Tuple, List

from batting_order_analysis.simulation import Simulation
from batting_order_analysis.lineup import Lineup
from batting_order_analysis.player import Player

def get_lineup(lineup_filename: Optional[str]) -> Lineup:
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
        Runs the simulation
    '''

    start = timeit.default_timer()

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
