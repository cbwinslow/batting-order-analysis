'''
    File: Lineup.py
    Author: Drew Scott
    Description: Implementation of a lineup using Players
'''

from Player import Player

class Lineup:
    def __init__(self):
        self.players = []
        self.game_outcomes = []

    def add_player(self, player:Player) -> None:
        '''
            Adds a player to the lineup
        '''
        self.players.append(player)
    
    def get_player(self, first_name:str, last_name:str) -> Player:
        '''
            Returns the player in the lineup who matches first and last name
        '''
        for player in self.players:
            if player.player_info['first_name'] == first_name and player.player_info['last_name'] == last_name:
                return player

        return None

    def set_pa_outcomes(self, outcome_filename:str, sims_per_order:int) -> None:
        '''
            Generates the PA outcomes for each player for each game that will be simulated
            Stores this information both in the corresponding Player instance and in this Lineup instance
        '''

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
