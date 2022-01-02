'''
    File: simulation.py
    Author: Drew Scott
    Description: 
'''

from statistics import mean
import random
from typing import List, Tuple, Optional, Dict
import heapq
import itertools
from multiprocessing import Pool
from functools import partial
import tqdm # type: ignore

from .lineup import Lineup

class Simulation:
    '''
        TODO
    '''

    def __init__(self, sims_per_order: int, lineup: Lineup):
        self.sims_per_order = sims_per_order
        self.lineup = lineup

    def run_sim(self):
        '''
            Simulates per_order games for each possible batting order
        '''

        orders = list(itertools.permutations(range(9)))

        # run the simulation
        avg_runs_per_order = []
        # TODO: 10?
        with Pool(10) as pool:
            avg_runs_per_order = list(tqdm.tqdm( \
                pool.imap(partial(self.sim_order), orders), \
                total=len(orders)))

        # aggregate data
        avg_runs_indexes = [(runs, i) for i, runs in enumerate(avg_runs_per_order)]
        best_five = heapq.nlargest(5, avg_runs_indexes)
        worst_five = heapq.nsmallest(5, avg_runs_indexes)

        best_runs = max(avg_runs_per_order)
        worst_runs = min(avg_runs_per_order)
        avg_runs = mean(avg_runs_per_order)

        # print results
        print(f'\nTotal games simulated: {len(orders) * self.sims_per_order:,}')
        print(f'Games simulated per order: {self.sims_per_order:,}')
        print(f'Total orders simulated: {len(orders):,}')
        print()
        print(f'Max runs for sim: {best_runs}')
        print(f'Avg. runs for sim: {avg_runs:.2f}')
        print(f'Min runs for sim: {worst_runs}\n')

        # TODO: only do this when user specifies lineup
        if True:
            print('Order of interest:')
            print(f'Average runs for order: {avg_runs_per_order[0]}')
            for ind in orders[0]:
                print(f'\t{str(self.lineup.players[ind])}')

        print('\nTop 5 batting orders:')
        for rank in range(len(best_five)):
            order = orders[best_five[rank][1]]
            print(f'{str(rank+1)}) Average runs for order: {best_five[rank][0]}')
            for ind in order:
                print(f'\t{str(self.lineup.players[ind])}')
            print()

        print('Bottom 5 batting orders:')
        for rank in range(len(worst_five)-1, -1, -1):
            order = orders[worst_five[rank][1]]
            print(f'{str(len(orders) - rank)}) Average runs for order: {worst_five[rank][0]}')
            for ind in order:
                print(f'\t{str(self.lineup.players[ind])}')
            print()

    def sim_order(self, order: List[int]) -> float:
        '''
            Simulates per_order games for the given order
        '''

        tot_runs_order = 0
        for game_num in range(self.sims_per_order):
            leadoff = 0
            thru_order = 0
            game_outcomes = self.lineup.game_outcomes[game_num]
            for _ in range(9):
                runs, leadoff, thru_order = self.sim_inning(\
                    game_outcomes, leadoff, thru_order, order)

                tot_runs_order += runs

        avg_runs_order = tot_runs_order / self.sims_per_order

        return avg_runs_order

    @staticmethod
    def new_batter(cur_batter: int, thru_order: int) -> Tuple[int, int]:
        '''
            Returns the index of the next batter
        '''
        if cur_batter == 8:
            return 0, thru_order+1

        return cur_batter + 1, thru_order

    def sim_inning(
            self,
            generated_outcomes: Dict[int, List[int]],
            leadoff: int,
            thru_order: int,
            order: List[int],
        ) -> Tuple[int, int, int]:

        '''
            Simulates an inning of play
            Returns the number of runs scored in the inning and what batter will lead off next inning
        '''

        runs = 0
        outs = 0
        runners: List[Optional[Player]] = [None, None, None]

        cur_batter_pos = leadoff
        cur_batter = order[cur_batter_pos]
        cur_player = self.lineup.players[cur_batter]

        while outs < 3:
            # TODO: simulate stealing and other mid-PA events

            # there aren't enough outcomes for this batter, so guarantee an out
            # TODO: simulate PA -- should do this in Player class so it holds constant across sims
            if len(generated_outcomes[cur_batter]) <= thru_order:
                outs += 1
                cur_batter_pos, thru_order = self.new_batter(cur_batter_pos, thru_order)
                cur_batter = order[cur_batter_pos]
                continue

            outcome = generated_outcomes[cur_batter][thru_order]

            # TODO: make it so the bit shifting can handle changes to outcomes
            # and how much info is encoded in outcome int
            if (8 << 2 <= outcome < 12 << 2) or (4 << 2 <= outcome < 5 << 2):
                # out on fly, ground, ld, or popup or strikeout

                # need to account for sacrficies and double/triple plays
                outs += 1
                if outs == 3:
                    # end of inning, no sacs possible
                    break

                # TODO: actually find rates
                if outcome == 'b_out_fly':
                    # sacrifice rates
                    third_scores = 0.2
                    second_advances = 0.15
                    if outcome & 0b11 == 2:
                        # hit to right
                        second_advances = 0.3
                    elif outcome & 0b11 == 0:
                        # hit to left
                        second_advances = 0.05

                    sac_rand = random.random()
                    if sac_rand < third_scores and runners[2] is not None:
                        runs += 1
                        runners = [runners[0], runners[1], None]

                    if sac_rand < second_advances and runners[1] is not None and runners[2] is None:
                        runners = [runners[0], None, runners[1]]

                elif outcome == 'b_out_ground':
                    sac_rate = 0.001
                    double_play_rate = 0.3

                    if random.random() < sac_rate:
                        runs += int(runners[2] is not None)
                        runners = [None, runners[0], runners[1]]
                    elif random.random() < double_play_rate:
                        outs += int(runners[0] is not None)
                        runners = [None, runners[1], runners[2]]

                        if outs >= 3:
                            break

            elif 5 << 2 <= outcome < 8 << 2:
                # walk, catcher interference, or hbp
                # right now treating all these the same:
                # just cascade runner advancements along starting with the batter
                if runners[0] is not None:
                    if runners[1] is not None:
                        if runners[2] is not None:
                            # runners on first, second, and third
                            runs += 1
                            runners = [cur_player, runners[0], runners[1]]

                        else:
                            # runners on first and second
                            runners = [cur_player, runners[0], runners[1]]
                    else:
                        # runner on first, maybe third
                        runners = [cur_player, runners[0], runners[2]]
                else:
                    # no runner on first
                    runners = [cur_player, runners[1], runners[2]]

            # TODO: account for direction on hits for base runner advancements
            elif 0 << 2 <= outcome < 1 << 2:
                # on a single, runners on second and third score, and runner on first goes to second
                runs += int(runners[1] is not None) + int(runners[2] is not None)
                runners = [cur_player, runners[0], None]

            elif 1 << 2 <= outcome < 2 << 2:
                # on a double, a runner on first advances to third, batter to second, and others score
                runs += int(runners[1] is not None) + int(runners[2] is not None)
                runners = [None, cur_player, runners[0]]

            elif 2 << 2 <= outcome < 3 << 2:
                # all runners score, batter to third
                runs += int(runners[0] is not None) + int(runners[1] is not None) + \
                    int(runners[2] is not None)
                runners = [None, None, cur_player]

            elif 3 << 2 <= outcome < 4 << 2:
                # all runners and batter score
                runs += int(runners[0] is not None) + int(runners[1] is not None) + \
                    int(runners[2] is not None) + 1
                runners = [None, None, None]
            else:
                raise Exception(f"Invalid outcome name: {outcome}")

            cur_batter_pos, thru_order = self.new_batter(cur_batter_pos, thru_order)
            cur_batter = order[cur_batter_pos]
            cur_player = self.lineup.players[cur_batter]

        cur_batter_pos, thru_order = self.new_batter(cur_batter_pos, thru_order)

        return runs, cur_batter_pos, thru_order

