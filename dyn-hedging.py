import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

u = [1.1, 1.05, 1.15]
d = [1/(1.1), 1/(1.05), 1/(1.15)]

asset_price = [250, 120, 130]
STRIKE = 270

def calc_inc_price(i):
    return asset_price[i] * u[i]

def calc_dec_price(i):
    return asset_price[i] * d[i]

a = np.zeros(shape=(4,2))
a[0][0] = calc_inc_price(1)
a[0][1] = calc_inc_price(2)
a[1][0] = calc_inc_price(1)
a[1][1] = calc_dec_price(2)
a[2][0] = calc_dec_price(1)
a[2][1] = calc_inc_price(2)
a[3][0] = calc_dec_price(1)
a[3][1] = calc_dec_price(2)

print a

b = np.zeros(4)
b[0] = max(calc_inc_price(0) - STRIKE, 0)
b[1] = max(calc_inc_price(0) - STRIKE, 0)
b[2] = max((calc_dec_price(0) - STRIKE), 0)
b[3] = max((calc_dec_price(0) - STRIKE), 0)

print b

w = np.zeros(2)
