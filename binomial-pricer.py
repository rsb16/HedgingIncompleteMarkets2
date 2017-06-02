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
num_assets = 3

u = np.exp(sigma * np.sqrt(DT))
d = np.exp((-1) * sigma * np.sqrt(DT))
end_prices = np.zeros(DTs+1)

start_prices_assets = [5, 6, 1]
assets_sigma = [0.32, 0.12, 0]
asset_us = np.zeros(num_assets)
asset_ds = np.zeros(num_assets)
end_prices_assets = np.zeros(shape=(num_assets, DTs + 1))

print_whole_tree = True


def set_us():
    global asset_us
    for i in range(len(asset_us) - 1):
        # asset_us[i] = np.exp(assets_sigma[i] * np.sqrt(DT))
        asset_us[i] = 1.1
    asset_us[len(asset_us) - 1] = INTEREST

def set_ds():
    global asset_ds
    for i in range(len(asset_ds) - 1):
        # asset_ds[i] = np.exp((-1) * assets_sigma[i] * np.sqrt(DT))
        asset_ds[i] = 1.0/1.1
    asset_ds[len(asset_ds) - 1] = INTEREST

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
        end_prices_assets[len(end_prices_assets) - 1][j] = find_bond_value(DTs)

def find_bond_value(t):
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
            prev_end_prices_assets[len(temp_asset_prices) - 1] = find_bond_value(t - 1)
            price = (calc_incomplete_price(temp_asset_prices, temp_end_prices, (t - 1)))
            prev_end_prices[i] = price
        # print(prev_end_prices)
        # print(prev_end_prices_assets)
        # print("--------------")
        return find_value_incomplete(prev_end_prices_assets, prev_end_prices, (t-1))

def calc_incomplete_price(end_prices_assets, end_prices, day):
    arrays = np.zeros(shape=(num_assets + 1, day + 1))
    for i in range(num_assets):
        for j in range(2):
            arrays[i][j] = end_prices_assets[i][j]
    for j in range(2):
        arrays[num_assets][j] = end_prices[j]
    perms = list(itertools.product(*arrays))
    A = np.zeros(shape=(pow(2, num_assets + 1), num_assets))
    for i in range(len(A)):
        for j in range(len(A[i])):
            perm = perms[i]
            A[i][j] = perm[j]
    b = np.zeros(pow(2, num_assets + 1))
    for i in range(len(b)):
        b[i] = perms[i][len(perms[i]) - 1]

    vals = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)),np.transpose(b))
    # print ("decisions: ", vals)
    value = 0
    for i in range(len(vals) - 1):
        value += end_prices_assets[i][0] * asset_us[i] * vals[i]
    value += vals[len(vals) - 1] * find_bond_value(day - 1)
    return value


set_finals()
set_us()
set_ds()
set_finals_assets()
val = find_value_incomplete(end_prices_assets, end_prices, DTs)
print ("our val: " + str(val))
val = find_value_optimal(end_prices, DTs)
print ("optimal val: " + str(val))
