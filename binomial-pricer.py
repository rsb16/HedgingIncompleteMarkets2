import numpy as np
import sys
import itertools

assert(len(sys.argv) == 7)

sigma = float(sys.argv[1])
DT = float(sys.argv[2])
DTs = int(sys.argv[3])
ANNUAL_INTEREST = (float(sys.argv[4]))
# INTEREST over the period DT.
INTEREST = 1 + (ANNUAL_INTEREST / (1.0 / DT))
start_price = float(sys.argv[5])
STRIKE = float(sys.argv[6])
num_assets = 2
num_trials = 1000

u = np.exp(sigma * np.sqrt(DT))
d = np.exp((-1) * sigma * np.sqrt(DT))
end_prices = np.zeros(DTs+1)

start_prices_assets = [60, 3]
assets_sigma = [0.3, 0.25]
rhos = [0.45, 0.56]
asset_us = np.zeros(num_assets)
asset_ds = np.zeros(num_assets)
norms_0 = np.zeros(num_trials)
norms_assets = np.zeros(shape=(num_assets, num_trials))
end_prices_assets = np.zeros(shape=(num_assets, DTs + 1))

print_whole_tree = False
with_bond = True

def set_norms():
    for i in range(num_trials):
        norms_0[i] = np.random.randn()
    for i in range(len(norms_assets)):
        for j in range(len(norms_assets[0])):
            norms_assets[i][j] = rhos[i] * norms_0[j] + np.sqrt(1 - rhos[i]) * np.random.randn()

def set_us():
    global asset_us
    for i in range(len(asset_us) - 1):
        asset_us[i] = np.exp(assets_sigma[i] * np.sqrt(DT))
    if (with_bond):
        asset_us[len(asset_us) - 1] = INTEREST
    else:
        asset_us[len(asset_us) - 1] = np.exp(assets_sigma[len(asset_us) - 1] * np.sqrt(DT))

def set_ds():
    global asset_ds
    for i in range(len(asset_ds) - 1):
        asset_ds[i] = np.exp((-1) * assets_sigma[i] * np.sqrt(DT))
    if (with_bond):
        asset_ds[len(asset_ds) - 1] = INTEREST
    else:
        asset_ds[len(asset_us) - 1] = np.exp((-1) * assets_sigma[len(asset_us) - 1] * np.sqrt(DT))

def find_value_optimal(end_prices, day):
    # end_prices[1] = Cu
    # end_prices[0] = Cd
    if len(end_prices) == 2:
        if (print_whole_tree and day == 1):
            print (end_prices)
        return ((1.0)/(INTEREST)) * ((((INTEREST) - d) / (u - d)) * end_prices[1] + ((u - (INTEREST))/(u - d)) * end_prices[0])
    else:
        if (print_whole_tree):
            print (end_prices)
        prices = np.zeros(len(end_prices) - 1)
        for i in range(len(end_prices) - 1):
            temp = [end_prices[i], end_prices[i+1]]
            prices[i] = find_value_optimal(temp, (day))
        return find_value_optimal(prices, (day - 1))

def set_finals():
    global end_prices
    for i in range(len(end_prices)):
        end_prices[i] = max(start_price * (pow(u, i + 1)) * (pow(d, len(end_prices) - i)) - STRIKE, 0)

def set_finals_assets():
    global end_prices_assets
    for i in range(len(end_prices_assets) - 1):
        for j in range(len(end_prices_assets[i])):
            end_prices_assets[i][j] = start_prices_assets[i] * (pow(asset_us[i], j + 1)) * (pow(asset_ds[i], len(end_prices_assets[i]) - j))
    for j in range(len(end_prices_assets[len(end_prices_assets) - 1])):
        if (with_bond):
            end_prices_assets[len(end_prices_assets) - 1][j] = find_bond_value(DTs)
        else:
            end_prices_assets[len(end_prices_assets) - 1][j] = start_prices_assets[len(end_prices_assets) - 1] * (pow(asset_us[len(end_prices_assets) - 1], j + 1)) * (pow(asset_ds[len(end_prices_assets) - 1], len(end_prices_assets[len(end_prices_assets) - 1]) - j))

def find_bond_value(t):
    # print("bond value at t: " + str(t) + " = " + str(start_prices_assets[len(start_prices_assets) - 1] * pow(asset_us[len(asset_us) - 1], t)))
    return start_prices_assets[len(start_prices_assets) - 1] * pow(asset_us[len(asset_us) - 1], t)

def find_value_incomplete(end_prices_assets, end_prices, t):
    # print (end_prices_assets)
    # print (end_prices)
    if (len(end_prices) == 2):
        if (print_whole_tree):
            print (end_prices)
        return calc_incomplete_price(end_prices_assets, end_prices, t)
    else:
        # print (end_prices_assets)
        # print (end_prices)
        # print ("-------#------")
        prev_end_prices = np.zeros(len(end_prices) - 1)
        prev_end_prices_assets = np.zeros(shape=(num_assets, len(end_prices_assets[0]) - 1))
        if (print_whole_tree):
            print (end_prices)
        for i in range(len(end_prices) - 1):
            temp_asset_prices = np.zeros(shape=(num_assets, 2))
            #Gets the elements pairwise
            for j in range(len(temp_asset_prices)):
                for k in range(2):
                    temp_asset_prices[j][k] = end_prices_assets[j][k + i]
            temp_end_prices = [end_prices[i], end_prices[i+1]]
            for j in range(len(temp_asset_prices) - 1):
                prev_end_prices_assets[j][i] = temp_asset_prices[j][0] * asset_us[j]
            if (with_bond):
                prev_end_prices_assets[len(temp_asset_prices) - 1] = find_bond_value(t - 1)
            else:
                prev_end_prices_assets[len(temp_asset_prices) - 1] = temp_asset_prices[len(temp_asset_prices) - 1][0] * asset_us[len(temp_asset_prices) - 1]
                # print("prev_end_prices_assets : " + str(prev_end_prices_assets))
            price = (calc_incomplete_price(temp_asset_prices, temp_end_prices, (t - 1)))
            prev_end_prices[i] = price
        # print("--------------")
        return find_value_incomplete(prev_end_prices_assets, prev_end_prices, (t-1))

def calc_incomplete_price(end_prices_assets, end_prices, day):
    set_norms()
    A = np.zeros(shape=(num_trials, num_assets))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if ((norms_assets[j][i]) > 0):
                # end_prices_assets = [ass_1[L,H], ass_2[L,H]]
                A[i][j] = end_prices_assets[j][1]
            else:
                A[i][j] = end_prices_assets[j][0]
    b = np.zeros(num_trials)
    for i in range(len(b)):
        if (norms_0[i] > 0):
            b[i] = end_prices[1]
        else:
            b[i] = end_prices[0]

    vals = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)),np.transpose(b))
    # diff = 0
    value = 0
    # for perm in perms:
    #     total = 0
    #     for asset in range(len(vals)):
    #         total += vals[asset] * perm[asset]
    #     diff = max(abs(perm[len(perms[0]) - 1] - total), diff)
    for i in range(len(vals) - 1):
        # print("multiplying: " + str(end_prices_assets[i][0] * asset_us[i]) + " and " + str(vals[i]) + " = " + str(end_prices_assets[i][0] * asset_us[i] * vals[i]))
        value += end_prices_assets[i][0] * asset_us[i] * vals[i]
    if (with_bond):
        value += vals[len(vals) - 1] * find_bond_value(day)
    else:
        # print("multiplying: " + str(end_prices_assets[len(vals) - 1][0] * asset_us[len(vals) - 1]) + " and " + str(vals[len(vals) - 1]) + " = " + str(end_prices_assets[len(vals) - 1][0] * asset_us[len(vals) - 1] * vals[len(vals) - 1]))
        value += vals[len(vals) - 1] * end_prices_assets[len(vals) - 1][0] * asset_us[len(vals) - 1]
    return value


set_finals()
set_us()
set_ds()
set_finals_assets()
val_1 = find_value_incomplete(end_prices_assets, end_prices, DTs)
print (val_1)
val_2 = find_value_optimal(end_prices, DTs)
print (val_2)
