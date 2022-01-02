'''
    File: lineup.py
    Author: Drew Scott
    Description: Implementation of a lineup using Players
'''

import pkg_resources
from typing import List, Optional
import random

from .player import Player

class Lineup:
    '''
        Stores information about the current lineup
        Contains methods to run simulations on the lineup
    '''

    data_directory = 'data/'
    lineups_directory = 'teams/'
    outcomes_directory = 'pa_outcomes/'
    stats_filename = 'stats.csv'

    def __init__(self):
        self.players: List[Player] = []
        self.game_outcomes = []

        # TODO: good place to put this?
        Player.set_metadata(Lineup.stats_filename)

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
            Sets the players in the lineup file to this lineup, in the order listed in the file
        '''

        lineup_filepath = Lineup.data_directory + Lineup.lineups_directory + lineup_filename
        stats_filepath = Lineup.data_directory + Lineup.stats_filename

        # get the players specified in the input file
        raw_players = pkg_resources.resource_stream(__name__, lineup_filepath).read().decode().split('\n')[:-1]
        player_names = []
        for player in raw_players:
            first, last = player.split()
            player_names.append(f'{last},{first}')

        # read the player data from the stats csv
        stats = pkg_resources.resource_stream(__name__, stats_filepath).read().decode(encoding='utf-8-sig')
        stat_lines = stats.split('\n')

        players: List[Optional[Player]] = [None] * 9 
        for line in stat_lines[1:]:
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

        # TODO: maybe make this a static class variable?
        stats_filepath = Lineup.data_directory + Lineup.stats_filename

        stats = pkg_resources.resource_stream(__name__, stats_filepath).read().decode(encoding='utf-8-sig')
        stat_lines = stats.split('\n')

        # get all the players
        players = []
        for line in stat_lines[1:]:
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
            outcome_filepath = Lineup.data_directory + Lineup.outcomes_directory + outcome_filename
            raw_outcomes = pkg_resources.resource_stream(__name__, outcome_filepath).read().decode().split('\n')[:-1]
            for player_outcome in raw_outcomes:
                first_name, last_name = player_outcome.split(':')[0].split()
                player = self.get_player(first_name, last_name)

                player.set_pa_outcomes(player_outcome.split(':')[1].split(','))

        for game_num in range(sims_per_order):
            game_pas = {}
            for i, player in enumerate(self.players):
                game_pas[i] = player.pa_outcomes[game_num]

            self.game_outcomes.append(game_pas)
