import numpy as np
import random
from test_denault import get_all_weights, calculate
from binomial_pricer import get_weights, calc
from find_destination_date import get_last_prices

INTEREST = 1.015
STRIKE = 55.7
DT = 1.0/365.0
num_days = 5
num_trials = 1
start_prices = [50, 34, 20, 1]
vols = [0.2, 0.29, 0.14, 0.11]
rhos = [1, 0.45, 0.22, 1]

norms_untraded = np.zeros(shape=(num_trials, num_days))
norms = np.zeros(shape=(num_trials * (len(start_prices) - 1), num_days))
prices = np.zeros(shape=(num_trials * len(start_prices), num_days))
prices_flights = np.zeros(num_days)
num_real_assets = 6
prices_assets = np.zeros(shape=(num_real_assets, num_days))
REAL_DATA = True

def set_untraded_norms():
    for i in range(num_trials):
        for j in range(num_days):
            norms_untraded[i][j] = np.random.randn();

def set_norms():
    for i in range(num_trials * (len(start_prices) - 1)):
        for j in range(num_days):
            norms[i][j] = rhos[(i + 1) % (len(start_prices) - 1)] * norms_untraded[i / (len(start_prices) - 1)][j] + np.sqrt(1 - rhos[(i + 1) % len(start_prices)]) * np.random.randn()

def initial_prices():
    for i in range(num_trials * len(start_prices)):
        prices[i][0] = start_prices[i % len(start_prices)]

def set_prices_untraded():
    if (not REAL_DATA):
        for i in range(num_trials):
            for j in range(1, num_days):
                if (norms_untraded[i][j] > 0):
                    prices[i * len(start_prices)][j] = prices[i * len(start_prices)][j - 1] * np.exp(vols[0] * np.sqrt(DT))
                else:
                    prices[i * len(start_prices)][j] = prices[i * len(start_prices)][j - 1] * np.exp(-1 * vols[0] * np.sqrt(DT))
    else:
        global prices_flights
        prices_flights = get_last_prices(num_days)
        global STRIKE
        STRIKE = prices_flights[0]

def gen_paths():
    #print norms
    #print prices
    for i in range(1, len(prices)):
        for j in range(1, len(prices[0])):
            if (i % len(start_prices) == 0):
                continue
            if ((i + 1) % len(start_prices) == 0):
                prices[i][j] = prices[i][j - 1] * np.exp((INTEREST - 1) * DT)
            elif (norms[i  - (i / len(start_prices))][j] > 0):
                prices[i][j] = prices[i][j - 1] * np.exp(vols[i % len(start_prices)] * np.sqrt(DT))
            else:
                prices[i][j] = prices[i][j - 1] * np.exp( - 1 * vols[i % len(start_prices)] * np.sqrt(DT))


def triangle(n):
    return n * (n + 1) / 2

def get_weights_for_day_and_position(day, count):
    #count is the number of times up and down
    #day is the day number
    #check but it should be ok
    return weights[triangle(day-1) + count - 1]

def set_prices_assets():
    with open("data/real-asset-data-past-5/gold.txt") as f:
        count = 0
        for line in f:
            prices_assets[0][count] = float(line)
            count += 1
    with open("data/real-asset-data-past-5/CL1.txt") as f:
        count = 0
        for line in f:
            prices_assets[1][count] = float(line)
            count += 1
    with open("data/real-asset-data-past-5/CO1.txt") as f:
        count = 0
        for line in f:
            prices_assets[2][count] = float(line)
            count += 1
    with open("data/real-asset-data-past-5/AAL.txt") as f:
        count = 0
        for line in f:
            prices_assets[3][count] = float(line)
            count += 1
    with open("data/real-asset-data-past-5/BA.txt") as f:
        count = 0
        for line in f:
            prices_assets[4][count] = float(line)
            count += 1
    with open("data/real-asset-data-past-5/AIR.txt") as f:
        count = 0
        for line in f:
            prices_assets[5][count] = float(line)
            count += 1


#   simulate a price path for each asset - done.
def sim():
    all_profits = 0
    n = 1
    if (not REAL_DATA):
        for i in range(n):
            set_untraded_norms()
            set_norms()
            initial_prices()
            set_prices_untraded()
            gen_paths()

            #   at each stage invest the correct amount in each asset

            #One trial
            count_prev = 0
            count_curr = 0
            total_prof = 0
            for k in range(1, num_days):
                profit_day = 0
                count_prev = count_curr
                #fix for multiple trials the 0.
                if (prices[0][k] > prices[0][k - 1]):
                    count_curr += 1
                else:
                    count_curr = max(0, count_curr - 1)
                weights_prev = get_weights_for_day_and_position(k - 1, count_prev)
                for j in range(len(start_prices) - 1):
                    profit_day += weights_prev[j] * (prices[j + 1][k] - prices[j + 1][k - 1])
                total_prof += profit_day
            # print max(prices[0][num_days - 1] - STRIKE, 0)
            # sell the stock we have
            ws = get_weights_for_day_and_position(num_days - 1, (num_days - (count_curr) - 1))
            port = 0
            for j in range(len(start_prices) - 1):
                port += ws[j] * prices[j + 1][num_days - 1]
            all_profits += (total_prof + port - max(prices[0][num_days - 1] - STRIKE, 0))
    else:
        count = 0
        port = 0
        ws = get_weights_for_day_and_position(0, count)
        for day in range(1, num_days):
            if (prices_flights[day] > prices_flights[day - 1]):
                count += 1
            else:
                count = max(count - 1, 0)
            profit_day = 0
            for i in range(len(ws)):
                profit_day += ws[i] * (prices_assets[i][day] - prices_assets[i][day - 1])
            port += profit_day
            ws = get_weights_for_day_and_position(day, count)
        for i in range(len(ws)):
            port += ws[i] * prices_assets[i][num_days - 1]
        print "Port is: ", port
        print "Payoff is: ", max(prices_flights[num_days - 1] - STRIKE, 0)
        # print "val: ", val
        # print "prices: ", prices_flights
        # print "prices_assets: ", prices_assets
        # print "weights: ", weights
        # print "Hello world"
    #   work out final profit and loss
    print all_profits / n

#option_val = calculate()
#weights = get_all_weights()
#sim()
if (REAL_DATA):
    set_prices_untraded()
    set_prices_assets()
val = calc()
weights = get_weights()
sim()
