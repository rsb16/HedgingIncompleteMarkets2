#!/usr/bin/env python

import copy
import sys
import json
import urllib2
import datetime

url = 'https://api.sandbox.amadeus.com/v1.2/flights/extensive-search'

origin = 'LON'
dests = ['STO', 'NYC', 'AMS', 'IST', 'PAR', 'ROM', 'MIL', 'NCE', 'ALC', 'BUD', 'VIE', 'GVA', 'BCN', 'NAP']

for dest in dests:
    query = '?'
    query += 'apikey={}&'.format(sys.argv[1])
    query += 'origin={}&'.format(origin)
    query += 'destination={}&'.format(dest)
    # query += 'departure_date={}&'.format(args[3])
    query += 'one-way=true&direct=true'

    with open('flight-data-{}.json'.format(dest), 'r') as f:
        current_data = json.loads(f.read())

        base_points = json.loads(urllib2.urlopen(url + query).read())

        data_ = copy.deepcopy(base_points)

        del data_['results']
        data_['now'] = str(datetime.datetime.now())
        for i in base_points['results']:
            data_['results'] = copy.deepcopy(i)
            current_data.append(copy.deepcopy(data_))

    with open('flight-data-{}.json'.format(dest), 'w') as f:
        f.write(json.dumps(current_data, indent=2))
