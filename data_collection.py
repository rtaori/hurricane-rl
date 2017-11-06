import requests
import re
import pandas

base_url = 'http://weather.unisys.com/hurricane'
pattern = re.compile('href="(.*)/track.dat"')

basins = ['atlantic', 'w_pacific', 'e_pacific']
years = range(1950, 2018)

for basin in basins:
    for year in years:

        try:
            r = requests.get(base_url + '/{0}/{1}/index.php'.format(basin, year))

            hurricanes = pattern.findall(r.text)
            for hurricane in hurricanes:

                r = requests.get(base_url + '/{0}/{1}/{2}/track.dat'.format(basin, year, hurricane))
                
                relevant_data = r.text.split('\n')[2:-1]
                data = []
                for line in relevant_data:
                    ln = line.strip().split()[1:]
                    if len(ln) == 7:
                        ln[5] = ln[5] + ' ' + ln[6]
                        del ln[-1]
                    data.append(ln)

                pd = pandas.DataFrame(data)
                pd.to_csv('data/{0}-{1}-{2}.csv'.format(basin, year, hurricane.lower()), index=False, header=False)

        except requests.exceptions.ConnectionError as e:
            print('Could not fetch data: /{0}/{1}/{2}/track.dat'.format(basin, year, hurricane))