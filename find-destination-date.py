#!/usr/bin/env python

import json
import sys
destination = unicode(sys.argv[1])
date = unicode(sys.argv[2])

with open('flight-data-{}-test.json'.format(destination), 'r') as f:
    current_data = json.loads(f.read())

filtered_data = filter(
    lambda x:   'results' in x and
                x['results']['destination'] == destination and
                x['results']['departure_date'] == date,
    current_data
)

mapped_data = map(lambda x: x['results']['price'], filtered_data)

print(json.dumps(mapped_data, indent=2))
