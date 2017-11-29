class TemperatureDictionary:
    DECADES = ['0', '55-64', '65-74', '75-84', '85-94', '95-04', '05-12']
    SEASONS = ['winter', 'spring', 'summer', 'autumn']

    def __init__(self):
        self.temp_data = {d : {s: {} for s in self.SEASONS} for d in self.DECADES}

    def add(self, year, season, data_row):
        decade = self._get_decade(year)
        lat_lon_key = self._make_key(data_row)
        self.temp_data[decade][season][lat_lon_key] = data_row[2]

    def get(self, data_row):
        year, month = data_row[2], data_row[3]
        decade = self._get_decade(year)
        season = self._get_season(month)

        # NOTE: Should nudge the lat lon pair by 0.25 until a pair is found.
        # TODO: Need to implement this behavior.
        lat_lon_key = self._make_key(data_row)
        return self.temp_data[decade][season][lat_lon_key]

    def _get_decade(self, year):
        if year < 1955: return '0'
        if year < 1964: return '55-64'
        if year < 1974: return '65-74'
        if year < 1984: return '75-84'
        if year < 1994: return '85-94'
        if year < 2004: return '95-04'
        if year < 2013: return '05-12'
        else: return '0'

    def _get_season(self, month):
        """Assumes northern hemisphere."""
        if month in [4, 5, 6]: # April - June
            return 'spring'
        if month in [7, 8, 9]: # July - September
            return 'summer'
        if month in [10, 11, 12]: # October - December
            return 'autumn'
        if month in [1, 2, 3]: # January - March
            return 'winter'
        raise ValueError('Month {0} is not valid!'.format(month))

    def _make_key(self, data_row):
        return data_row[0] + "&" + data_row[1]
