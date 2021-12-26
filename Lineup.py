'''
    File: Lineup.py
    Author: Drew Scott
    Description:
'''

import random
from typing import List

class Player:
    # TODO: don't hardcode this, but try to keep as a static variable
    col_names = ['last_name', 'first_name', 'player_id', 'year', 'b_total_pa', 'b_single', 'b_double', 'b_triple', 'b_home_run', 'b_strikeout', 'b_walk', \
                'slg_percent', 'on_base_percent', 'on_base_plus_slg', 'b_catcher_interf', 'b_hit_by_pitch', 'b_out_fly', 'b_out_ground', 'b_out_line_drive', 'b_out_popup']

    pa_col_names = ['b_single', 'b_double', 'b_triple', 'b_home_run', 'b_strikeout', 'b_walk','b_catcher_interf', 'b_hit_by_pitch', 'b_out_fly', 'b_out_ground', 'b_out_line_drive', 'b_out_popup']

    def __init__(self, stat_line: str):
        self.player_info = self.get_info(stat_line)
        self.player_pa_probs = self.get_pa_probs(self.player_info)
        self.player_pa_thresholds = self.get_pa_thresholds()
        self.pa_outcomes = None

    def set_pa_outcomes(self, pa_outcomes:List[str]) -> None:
        self.pa_outcomes = [pa_outcomes] 
    
    def get_pa_outcome(self) -> str:
        '''
            Returns the name of the plate appearance outcome based on a randomly generated number
        '''
        rand_outcome = random.random()
        for out_threshold, outcome in zip(self.player_pa_thresholds, Player.pa_col_names):
            if rand_outcome < out_threshold:
                return outcome

        return Player.pa_col_names[-1]

    def generate_pa_outcomes(self, n_games:int) -> None:
        generated_outcomes = []

        for _ in range(n_games):
            game_outcomes = []
            for _ in range(10):
                outcome = self.get_pa_outcome()
                game_outcomes.append(outcome)

            generated_outcomes.append(game_outcomes)
        
        self.pa_outcomes = generated_outcomes

    def get_pa_thresholds(self) -> List[float]:
        '''
            Returns the upper threshold (between 0 and 1) for each outcome. Used to decide a plate appearance outcome using a random number
        '''
        thresholds = []

        thresholds.append(self.player_pa_probs[Player.pa_col_names[0]])
        for i, outcome in enumerate(Player.pa_col_names[1:], start=1):
            thresholds.append(thresholds[i-1] + self.player_pa_probs[outcome])

        return thresholds

    def get_pa_probs(self, player_info: dict) -> dict:
        '''
            Returns a dict containing only the the probability of each possible plate appearance outcomes for the input player
        '''
        probs = {}

        total_pa = player_info['b_total_pa']

        for outcome, outcome_count in player_info.items():
            if outcome not in Player.pa_col_names:
                continue

            p = outcome_count / total_pa
            probs[outcome] = p

        return probs

    def get_info(self, stats_line:str) -> dict:
        '''
            Returns a dict of all of the player's information
        '''
        d = {}

        stats_line = stats_line[: -1].split(',')
        stats = [int(s) if s.isnumeric() else s for s in stats_line]

        for col, stat in zip(Player.col_names, stats):
            d[col] = stat

        return d

    def __repr__(self):
        return f'{self.player_info["first_name"]} {self.player_info["last_name"]}: ' + \
               f'{self.player_info["on_base_percent"]} OBP, {self.player_info["slg_percent"]} SLG, {self.player_info["on_base_plus_slg"]} OPS'

class Lineup:
    def __init__(self):
        self.players = []
        self.game_outcomes = []

    def add_player(self, player:Player) -> None:
        self.players.append(player)
    
    def get_player(self, first_name:str, last_name:str) -> Player:
        for player in self.players:
            if player.player_info['first_name'] == first_name and player.player_info['last_name'] == last_name:
                return player

        return None

    def set_pa_outcomes(self, outcome_filename:str, sims_per_order:int) -> None:
        if outcome_filename is None:
            for player in self.players:
                player.generate_pa_outcomes(sims_per_order)

        else:
            with open(outcome_filename, 'r') as outcome_f:
                for line in outcome_f:
                    first_name, last_name = line.split(':')[0].split()
                    player = self.get_player(first_name, last_name) 
                    if player is None:
                        raise Exception(f'Player {first_name} {last_name} not found from PA outcome file')

                    player.set_pa_outcomes(line.split(':')[1][:-1].split(','))

        for game_num in range(sims_per_order):
            game_pas = {}
            for i, player in enumerate(self.players):
                game_pas[i] = player.pa_outcomes[game_num]

            self.game_outcomes.append(game_pas)
