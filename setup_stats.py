'''
    File: setup_stats.py
    Author: Drew Scott
    Description: Sets up stats.csv: a) removes Baseball Savant's formatted spaces and b) add R/L batting splits for each player
    Usage: python3 setup_stats.py <-ds if only deleting spaces>
'''

import sys
from typing import Tuple
from bs4 import BeautifulSoup
import requests
import pandas as pd

def get_rl_splits(playerid:str) -> Tuple[float, float]:
    url = 'https://baseballsavant.mlb.com/savant-player/' + playerid + '?stats=statcast-r-batting-mlb&playerType=batter'
    player_page = requests.get(url).content
    soup = BeautifulSoup(player_page, 'html.parser')

    shift_dividers = soup.find_all('div', {'id':'statcast_stats_shift'})
    if len(shift_dividers) >= 1:
        shift_table = shift_dividers[0].find_next_sibling('div').table
        df = pd.read_html(str(shift_table))[0]

        df = df[df['Season'] == 2021]
        total_pa = df['PA'].sum()
        right_pa = df[df['Bat Side'] == 'R']['PA'].sum()
        left_pa = df[df['Bat Side'] == 'L']['PA'].sum()

        return right_pa / total_pa, left_pa / total_pa
    else:
        # lazy assume righty if no data
        print(url)
        return 1, 0

def get_playerid(stat_line:str) -> str:
    splits = stat_line.split(',')
    return splits[1].lower() + '-' + splits[0].lower() + '-' + splits[2]

if __name__ == '__main__':
    # remove all spaces (baseball savant name formatting) and
    # put left and right hitting pct for each player
    updated_content = ''

    only_ds = False
    if '-ds' in sys.argv:
        only_ds = True

    with open('stats.csv', 'r', encoding='utf-8-sig') as stats_csv:
        col_names = stats_csv.readline()[:-1]
        col_names = col_names.replace(' ', '')
        col_names += 'bats_right_pct,bats_left_pct,\n'
        updated_content += col_names

        for i, line in enumerate(stats_csv):
            # remove spaces
            updated_line = line.replace(' ', '')

            if not only_ds:
                print(f'{i+1}/556   ', end='\r')
                # get batting side info
                playerid = get_playerid(updated_line)

                right, left = get_rl_splits(playerid)
                updated_line = f'{updated_line[:-1]}{right},{left},\n'

            updated_content += updated_line

    # write the new content back out
    with open('stats.csv', 'w', encoding='utf-8-sig') as stats_csv:
        stats_csv.write(updated_content)

    if not only_ds:
        print()
