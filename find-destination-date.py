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

mapped_data = map(lambda x: x['results']['price'], filtered_data)
all_prices = mapped_data
prices = []
returns = []

for i in range(len(all_prices)):
    prices.append(float(all_prices[i]))
    if (i > 0):
        returns.append(math.log(float(all_prices[i]) / float(all_prices[i - 1])))

avg_return = np.divide(sum(returns), len(returns))
sq_dev = 0
for i in range(len(returns)):
    sq_dev += pow((returns[i] - avg_return), 2)

sq_dev = np.divide(sq_dev, len(returns))
vol = math.sqrt(sq_dev)
print vol * math.sqrt(252)
plt.plot(prices)
#plt.show()
