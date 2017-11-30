import os
from collections import deque
import numpy as np

class TemperatureDictionary:

    DECADES = ['55-64', '65-74', '75-84', '85-94', '95-04', '05-12']
    SEASONS = ['winter', 'spring', 'summer', 'autumn']
    LAT_LON_NUDGE = [[0.25, 0], [-0.25, 0], [0, 0.25], [0, -0.25]]
    VALID_DECIMALS = [0.125, 0.375, 0.625, 0.875]

    def __init__(self):
        self.temp_data = {d : {s: {} for s in self.SEASONS} for d in self.DECADES}

    def initialize(self):
        for filename in os.listdir('data/ocean_temp'):
            if filename[-3:] != 'csv': continue
            file = list(open('data/ocean_temp/' + filename, 'r', encoding="latin-1"))
            x = [line.strip().split(',')[:3] for line in file[2:]]
            x = np.array(x)
            x = np.delete(x, np.where(x == ''), 0)

            metadata = filename.split('.')[0].split('_')[-1].split('-')
            year = int(metadata[0])
            season = metadata[-1]

            for x_i in x:
                self.add(year, season, x_i)

    def add(self, year, season, data_row):
        decade = self._get_decade(year)
        lat_lon_key = self._make_key(data_row)
        self.temp_data[decade][season][lat_lon_key] = data_row[2]

    def get(self, data_row):
        year, month = data_row[2], data_row[3]
        decade = self._get_decade(year)
        season = self._get_season(month)
        temp_dict = self.temp_data[decade][season]
        if not temp_dict: return 0.0 # Empty dictionary

        lat_lon_pair = (data_row[0], data_row[1])
        valid_lat_lon_pair = self._make_valid_lat_lon_pair(lat_lon_pair)
        entry_exists = lambda e: self._make_key(e) in temp_dict
        valid_lat_lon_pair = self._bfs_find(valid_lat_lon_pair, entry_exists)

        valid_key = self._make_key(valid_lat_lon_pair)
        return float(temp_dict[valid_key])

    def _get_decade(self, year):
        if year < 1955: return '55-64' # Treat pre-1955 as 1955-1964
        if year < 1964: return '55-64'
        if year < 1974: return '65-74'
        if year < 1984: return '75-84'
        if year < 1994: return '85-94'
        if year < 2004: return '95-04'
        if year < 2013: return '05-12'
        else: return '05-12' # Treat post-2012 as 2005-2012

    def _get_season(self, month):
        """Assumes northern hemisphere."""
        if month < 1 or month > 12:
            #raise ValueError('Month {0} is not valid!'.format(month))
            print('Month {0} is not valid! Adapting...'.format(month))
            month = (month % 12) + 1
        if month in [4, 5, 6]: # April - June
            return 'spring'
        if month in [7, 8, 9]: # July - September
            return 'summer'
        if month in [10, 11, 12]: # October - December
            return 'autumn'
        if month in [1, 2, 3]: # January - March
            return 'winter'

    def _make_key(self, data_row):
        return str(data_row[0]) + "&" + str(data_row[1])

    def _make_valid_lat_lon_pair(self, lat_lon_pair):
        def find_possible_value(value):
            return min([int(value) + dec for dec in self.VALID_DECIMALS],
                   key=lambda possible_value: abs(possible_value - value))
        lat, lon = lat_lon_pair[0], lat_lon_pair[1]
        return (find_possible_value(lat), find_possible_value(lon))

    def _get_neighbors(self, lat_lon_pair):
        lat, lon = lat_lon_pair[0], lat_lon_pair[1]
        for dlat, dlon in self.LAT_LON_NUDGE:
            yield (lat + dlat, lon + dlon)

    def _bfs_find(self, start, success_fn):
        visited = set()
        visited.add(start)
        q = deque()
        q.append(start)
        while q:
            u = q.popleft()
            if success_fn(u):
                return u
            for v in self._get_neighbors(u):
                if v not in visited:
                    q.append(v)
                    visited.add(v)
