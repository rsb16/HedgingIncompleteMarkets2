#!/usr/bin/env python

import json
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import re

destination = unicode(sys.argv[1])
date = unicode(sys.argv[2])

with open('flight-data-{}.json'.format(destination), 'r') as f:
    current_data = json.loads(f.read())

filtered_data = filter(
    lambda x:   'results' in x and
                x['results']['destination'] == destination and
                x['results']['departure_date'] == date,
    current_data
)

prices = []
mapped_data = map(lambda x: x['results']['price'], filtered_data)

for i in range(len(mapped_data)):
    prices.append(float(mapped_data[i]))
print prices
# plt.plot(prices)

def get_last_prices(num_days):
    return prices[-num_days:]

#plt.show()
