import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from scipy.stats import norm

assert(len(sys.argv) == 7)

sigma = float(sys.argv[1])
DT = float(sys.argv[2])
DTs = int(sys.argv[3])
INTEREST = (float(sys.argv[4]))
start_price = float(sys.argv[5])
STRIKE = float(sys.argv[6])
num_assets = 2

u = np.exp(sigma * np.sqrt(DT))
d = np.exp((-1) * sigma * np.sqrt(DT))

end_prices = np.zeros(DTs+1)

start_prices = [50, 90]
asset_sigma = [0.22, 0.12]
asset_us = [np.exp(asset_sigma[0] * np.sqrt(DT)), np.exp(asset_sigma[1] * np.sqrt(DT))]
asset_ds = [np.exp((-1) * asset_sigma[0] * np.sqrt(DT)), np.exp((-1) * asset_sigma[1] * np.sqrt(DT))]

end_prices_0 = np.zeros(DTs+1)
end_prices_1 = np.zeros(DTs+1)

def find_value_optimal(end_prices):
    # end_prices[1] = Cu
    # end_prices[0] = Cd
    if len(end_prices) == 2:
        return ((1.0)/(INTEREST)) * ((((INTEREST) - d) / (u - d)) * end_prices[1] + ((u - (INTEREST))/(u - d)) * end_prices[0])
    else:
        prices = np.zeros(len(end_prices) - 1)
        for i in range(len(end_prices) - 1):
            temp = [end_prices[i], end_prices[i+1]]
            prices[i] = find_value_optimal(temp)
        return find_value_optimal(prices)

def find_value_incomplete(end_prices_0, end_prices_1, end_prices):
    # end_prices[1] = Cu
    # end_prices[0] = Cd
    if len(end_prices_0) == 2:
        vals = calc_incomplete_price(end_prices_0, end_prices_1, end_prices)
        print vals
        return vals
    else:
        untraded = np.zeros(len(end_prices) - 1)
        prices_0 = np.zeros(len(end_prices_0) - 1)
        prices_1 = np.zeros(len(end_prices_1) - 1)
        for i in range(len(prices_0)):
            temp_0 = [end_prices_0[i], end_prices_0[i+1]]
            temp_1 = [end_prices_1[i], end_prices_1[i+1]]
            temp_untraded = [end_prices[i], end_prices[i+1]]
            vals = calc_incomplete_price(temp_0, temp_1, temp_untraded)
            # print 'temp_0: ', temp_0
            # print 'temp_1: ', temp_1
            # print 'temp_untraded: ', temp_untraded
            # print 'vals: ', vals
            # print 'i: ', i
            # print 'adding: ', (vals[0] * temp_0[0] * asset_ds[0] + vals[1] * temp_1[1] * asset_ds[1])
            untraded[i] = (vals[0] * temp_0[0] * asset_ds[0] + vals[1] * temp_1[1] * asset_ds[1])
            prices_0[i] = end_prices_0[i] * asset_ds[0]
            prices_1[i] = end_prices_1[i] * asset_ds[1]
        return find_value_incomplete(prices_0, prices_1, untraded)

def set_finals():
    global end_prices
    for i in range(len(end_prices)):
        end_prices[i] = max(start_price * (pow(u, i + 1)) * (pow(d, len(end_prices) - i)) - STRIKE, 0)

def set_end_prices_assets():
    global end_prices_0
    global end_prices_1
    for i in range(len(end_prices_0)):
        end_prices_0[i] = start_prices[0] * (pow(asset_us[0], i + 1)) * (pow(asset_ds[0], len(end_prices_0) - i))
        end_prices_1[i] = start_prices[1] * (pow(asset_us[1], i + 1)) * (pow(asset_ds[1], len(end_prices_1) - i))

def calc_incomplete_price(end_prices_0, end_prices_1, end_prices):


    perms = list(itertools.product(end_prices_0, end_prices_1, end_prices))

    A = np.zeros(shape=(pow(2, num_assets + 1), num_assets))
    for i in range(len(A)):
        for j in range(len(A[i])):
            perm = perms[i]
            A[i][j] = perm[j]
    b = np.zeros(pow(2, num_assets + 1))
    for i in range(len(b)):
        b[i] = perms[i][len(perms[i]) - 1]
    vals = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)),np.transpose(b))

    # still need to do something about the INTEREST rate?
    # value = 0
    # for i in range(len(vals)):
    #     value += vals[i] * start_price_assets[i]
    # return value
    return vals

set_finals()
set_end_prices_assets()
print end_prices
print end_prices_0
print end_prices_1
vals = find_value_incomplete(end_prices_0, end_prices_1, end_prices)
value = 0
for i in range(len(vals)):
    value += vals[i] * start_prices[i]
print 'our value: ', value
print 'optimal val: ', find_value_optimal(end_prices)
# print 'our val: ', find_value_incomplete()
