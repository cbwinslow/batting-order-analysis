'''
    File: lineup.py
    Author: Drew Scott
'''

import pkg_resources
from typing import List, Optional
import random

from .player import Player

class Lineup:
    '''
        Stores the players in a Lineup

        Public methods:
            set_players: sets players in lineup either randomly or based on input filename
            get_player: returns a Player instance who matches first_name and last_name from
                the players list
    '''

    def __init__(self):
        self.players: List[Player] = []
        self.random_lineup = False

    def set_players(self,
        lineup_filename: Optional[str] = None,
        lineup_names_arr: Optional[List[str]] = None
    ) -> None:
        '''
            Sets the players in this lineup
        '''

        # reset to default
        self.players = []
        self.random_lineup = False

        # add the players
        if lineup_names_arr is not None:
            self._generate_playername_arr_lineup(lineup_names_arr)

        elif lineup_filename is not None:
            self._generate_file_lineup(lineup_filename)

        else:
            # randomly generate players
            self._generate_random_lineup()
            self.random_lineup = True

        # display the players
        print("Players:")
        for player in self.players:
            print(player)
        print()

    def get_player(self, first_name: str, last_name: str) -> Player:
        '''
            Returns the player in the lineup who matches first and last name
        '''

        for player in self.players:
            if player.player_info['first_name'] == first_name \
                and player.player_info['last_name'] == last_name:
                return player

        raise Exception(f'Player {first_name} {last_name} not found from PA outcome file')

    def _generate_random_lineup(self) -> None:
        '''
            Generates a random lineup of 9 players drawn from stats_filename
        '''

        stats = pkg_resources.resource_stream(__name__, Player.stats_filepath).read().decode(encoding='utf-8-sig')
        stat_lines = stats.split('\n')

        # get all the players
        players = []
        for line in stat_lines[1:]:
            players.append(Player(line))

        # select 9 random players to add to the lineup
        player_indexes = self._get_nine(len(players))
        for index in player_indexes:
            self._add_player(players[index])

    # TODO: standardize player name formats to last_name,first_name
    def _generate_file_lineup(self, lineup_filename: str) -> None:
        '''
            Sets the players in the lineup file to this lineup, in the order listed in the file
            Player names in form: first_name last_name
        '''

        # TODO: what if lineup_filename not located in package contents (i.e. is user generated?)
        lineup_filepath = Player.data_directory + Player.lineups_directory + lineup_filename

        # get the players specified in the input file
        raw_players = pkg_resources.resource_stream(__name__, lineup_filepath).read().decode().split('\n')[:-1]
        player_names = []
        for player in raw_players:
            first, last = player.split()
            player_names.append(f'{last},{first}')

        self._generate_playername_arr_lineup(player_names)

    def _generate_playername_arr_lineup(self, lineup_names_arr: List[str]) -> None:
        '''
            Sets the players in the input name list to be the players in the lineup
            Player names in form: last_name,first_name
        '''

        # read the player data from the stats csv
        stats = pkg_resources.resource_stream(__name__, Player.stats_filepath).read().decode(encoding='utf-8-sig')
        stat_lines = stats.split('\n')

        players: List[Optional[Player]] = [None] * 9
        for line in stat_lines[1:]:
            splits = line.split(',')
            name = splits[0] + ',' + splits[1]

            if name in player_names:
                players[player_names.index(name)] = Player(line)

        for player in players:
            self._add_player(player)

        if len(self.players) != 9:
            raise Exception(f'Incorrect number of players: {len(lineup.players)}, {lineup.players}')

    def _add_player(self, player: Optional[Player]) -> None:
        '''
            Adds a player to the lineup
        '''

        if player is None:
            raise Exception('Tried adding None to lineup')

        self.players.append(player)

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
