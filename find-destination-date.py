#!/usr/bin/env python

import json
import sys

with open('flight-data.json.backup', 'r') as f:
    current_data = json.loads(f.read())

destination = unicode(sys.argv[1])
date = unicode(sys.argv[2])

filtered_data = filter(
    lambda x:   'results' in x and
                x['results'][0]['destination'] == destination and
                x['results'][0]['departure_date'] == date,
    current_data
)

mapped_data = map(lambda x: x['results'][0]['price'], filtered_data)

print(json.dumps(mapped_data, indent=2))
