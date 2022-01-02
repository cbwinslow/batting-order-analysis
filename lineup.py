'''
    File: lineup.py
    Author: Drew Scott
    Description: Implementation of a lineup using Players
'''

from typing import List, Optional
import random

from player import Player

class Lineup:
    '''
        Stores information about the current lineup
        Contains methods to run simulations on the lineup
    '''

    stats_filename = 'stats.csv'

    def __init__(self):
        self.players: List[Player] = []
        self.game_outcomes = []

    def add_player(self, player: Optional[Player]) -> None:
        '''
            Adds a player to the lineup
        '''
        if player is None:
            raise Exception('Tried adding None to lineup')

        self.players.append(player)

    def get_player(self, first_name: str, last_name: str) -> Player:
        '''
            Returns the player in the lineup who matches first and last name
        '''
        for player in self.players:
            if player.player_info['first_name'] == first_name \
                and player.player_info['last_name'] == last_name:
                return player

        raise Exception(f'Player {first_name} {last_name} not found from PA outcome file')

    def set_players(self, lineup_filename: str) -> None:
        '''
            Sets the players in the lineup file to this lineup, in the order they were listed in the file
        '''

        # get the players specified in the input file
        player_names = []
        with open(lineup_filename, 'r') as f_players:
            for line in f_players:
                first, last = line.split()
                player_names.append(f'{last},{first}')

        # read the player data from master file
        players: List[Optional[Player]] = [None] * 9 
        with open(Lineup.stats_filename, 'r', encoding='utf-8-sig') as stats_csv:
            col_names = stats_csv.readline().strip()[:-1].split(',')
            for line in stats_csv:
                splits = line.split(',')
                name = splits[0] + ',' + splits[1]

                if name in player_names:
                    players[player_names.index(name)] = Player(line)

        for player in players:
            self.add_player(player)

        if len(self.players) != 9:
            raise Exception(f'Incorrect number of players: {len(lineup.players)}, {lineup.players}')

    @staticmethod
    def _get_nine(total_count: int) -> List[int]:
        '''
            Returns a list of length 9 with unique indexes in the range of total_count
        '''
        nine: List[int] = []

        while len(nine) < 9:
            index = random.randint(0, total_count - 1)
            if index not in nine:
                nine.append(index)

        return nine

    def generate_random_lineup(self) -> None:
        '''
            Generates a random lineup of 9 players drawn from stats_filename
        '''

        players = []
        with open(Lineup.stats_filename, 'r', encoding='utf-8-sig') as stats_csv:
            # get all the players
            col_names = stats_csv.readline().strip()[:-1].split(',')
            for line in stats_csv:
                players.append(Player(line))

            # select 9 random players to add to the lineup
            player_indexes = self._get_nine(len(players))
            for index in player_indexes:
                self.add_player(players[index])

    def set_pa_outcomes(self, outcome_filename: Optional[str], sims_per_order: int) -> None:
        '''
            Generates the PA outcomes for each player for each game that will be simulated
            Stores this information both in the Player instance and in this Lineup instance
        '''

        if outcome_filename is None:
            for player in self.players:
                player.generate_pa_outcomes(sims_per_order)

        else:
            with open(outcome_filename, 'r') as outcome_f:
                for line in outcome_f:
                    first_name, last_name = line.split(':')[0].split()
                    player = self.get_player(first_name, last_name)

                    player.set_pa_outcomes(line.split(':')[1][:-1].split(','))

        for game_num in range(sims_per_order):
            game_pas = {}
            for i, player in enumerate(self.players):
                game_pas[i] = player.pa_outcomes[game_num]

            self.game_outcomes.append(game_pas)
