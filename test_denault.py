import numpy as np
from scipy import optimize
import random
import itertools
INTEREST = 1.015
STRIKE = 0
DT = 1.0/365.0
num_days = 5
num_trials = 10000
start_prices = [50, 34, 20, 1]
vols = [0.2, 0.29, 0.14, 0.11]
rhos = [1, 0.45, 0.22, 1]
num_assets = len(start_prices)
num_tradeables = len(start_prices) - 1
weights = np.zeros(shape=(num_tradeables, num_trials))
norms = np.zeros(shape=(num_trials, len(start_prices)))
end_prices = np.zeros(shape=(num_trials, len(start_prices)))
EPSILON = 0

all_weights = []

def set_norms():
    for i in range(len(norms)):
        for j in range(len(norms[0])):
            if (j == 0):
                norms[i][j] = np.random.randn()
            else:
                norms[i][j] = rhos[j] * norms[i][0] + np.sqrt(1 - rhos[j]) * np.random.randn()

def set_end_prices_non_final(start_prices, vals):
    for i in range(len(end_prices)):
        for j in range(len(end_prices[0]) - 1):
            if (norms[i][j] > 0):
                end_prices[i][j] = start_prices[j] * np.exp(vols[j] * np.sqrt(DT))
            else:
                end_prices[i][j] = start_prices[j] * np.exp(-1 * vols[j] * np.sqrt(DT))
            if (norms[i][0] > 0):
                end_prices[i][0] = vals[0]
            else:
                end_prices[i][0] = vals[1]
        end_prices[i][len(end_prices[0]) - 1] = start_prices[len(end_prices[0]) - 1] * np.exp((INTEREST - 1) * DT)

def get_all_weights():
    return all_weights

def set_end_prices(start_prices):
        for i in range(len(end_prices)):
            for j in range(len(end_prices[0]) - 1):
                if (norms[i][j] > 0):
                    end_prices[i][j] = start_prices[j] * np.exp(vols[j] * np.sqrt(DT))
                else:
                    end_prices[i][j] = start_prices[j] * np.exp(-1 * vols[j] * np.sqrt(DT))
            end_prices[i][0] = max(end_prices[i][0] - STRIKE, 0)
            end_prices[i][len(end_prices[0]) - 1] = start_prices[len(end_prices[0]) - 1] * np.exp((INTEREST - 1) * DT)

def set_weights():
    for i in range(num_tradeables):
        for j in range(num_trials):
            weights[i][j] = random.uniform(-5, 5)

def triangle(n):
    return n * (n + 1) / 2

def get_diag(matrix):
    diag = np.zeros(len(matrix))
    for i in range(len(matrix)):
        diag[i] = matrix[i][i]
    return diag

def get_upper_triang(matrix):
    values = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (j > i):
                values.append(matrix[i][j])
    return values

def get_second_order_terms(x):
    terms = []
    for i in range(triangle(len(start_prices) - 2)):
        terms.append(x[len(start_prices) + i])
    return terms

def get_second_order_terms_diag(x):
    terms = np.zeros(num_tradeables)
    for i in range(len(terms)):
        terms[i] = x[len(x) - len(start_prices) + 1 + i]
    return terms

def symmetrize(a):
    return a + a.T - numpy.diag(a.diagonal())

def regress(prev_prices):
    b = np.zeros(num_trials)
    for i in range(len(b)):
        port = 0
        for j in range(num_tradeables):
            port += weights[j][i] * end_prices[i][j + 1]
        b[i] = end_prices[i][0] - port
    A = np.zeros(shape=(num_trials, triangle(len(start_prices))))
    EPS = np.zeros(shape=(triangle(len(start_prices)), triangle(len(start_prices))))

    for i in range(len(EPS)):
        for j in range(len(EPS[0])):
            EPS[i][j] = EPSILON

    for i in range(len(A)):
        A[i][0] = 1
        weights_trial = np.zeros(num_tradeables)
        for j in range(num_tradeables):
            weights_trial[j] = weights[j][i]
            A[i][j + 1] = weights[j][i]
        shape = (num_tradeables, num_tradeables)
        higher_order_weights = (np.triu(np.array([x * y for (x,y) in list(itertools.product(weights_trial, weights_trial))]).reshape(shape)))
        higher_order_weights_diag = get_diag(np.array(np.diag(np.diag(higher_order_weights))))
        higher_order_weights_UT = get_upper_triang(np.array(np.subtract(higher_order_weights, np.diag(np.diag(higher_order_weights)))))
        for j in range(len(higher_order_weights_UT)):
            A[i][len(start_prices) + j] = higher_order_weights_UT[j]
        for j in range(len(higher_order_weights_diag)):
            A[i][- 1 * (len(higher_order_weights_diag)) + j] = higher_order_weights_diag[j]

    def loss(x):
        return abs(0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

    x = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(A), A), EPS)), np.transpose(A)), b)
    #print 'x: ', x
    c0 = x[0] # get the constant

    # get the single order terms
    single_order_terms = np.zeros(num_tradeables)

    for i in range(len(single_order_terms)):
        single_order_terms[i] = x[i + 1]

    second_order_terms_diag = get_second_order_terms_diag(x)

    second_order_terms = get_second_order_terms(x)

    H = np.zeros(shape=(num_tradeables, num_tradeables))

    for i in range(len(H)):
        for j in range(len(H[0])):
            if (i > j):
                H[i][j] = H[j][i]
            elif (i == j):
                H[i][j] = second_order_terms_diag[i]
            else:
                H[i][j] = second_order_terms.pop(0)

    c = single_order_terms
    opt = {'disp':False, 'maxiter':500}
    x0 = np.ones(num_tradeables)
    res_uncons = optimize.minimize(loss, x0, method='Nelder-Mead', options=opt)
    value = 0
    all_weights.append(res_uncons.x)
    for i in range(len(res_uncons.x)):
        value += res_uncons.x[i] * prev_prices[i + 1]
    return value

def asset_up(i):
    return np.exp(vols[i] * np.sqrt(DT))

def asset_down(i):
    return np.exp(-1 * vols[i] * np.sqrt(DT))

def bond_price():
    return np.exp((INTEREST - 1) * DT)


def get_all_end_prices(start_prices, day):
    end_prices = np.zeros(shape=(num_assets, day))
    for asset in range(num_assets):
        for i in range(day):
            end_prices[asset][i] = start_prices[asset] * pow(asset_up(asset), i) * pow(asset_down(asset), day - 1 - i)
    for i in range(day):
        end_prices[num_assets - 1][i] = start_prices[num_assets - 1] * pow(bond_price(), day - 1)
    return end_prices

def calculate():
    values = np.zeros(num_days - 1)
    for day in range(num_days, 1, -1):
        prev_prices = get_all_end_prices(start_prices, day - 1)
        for j in range(len(prev_prices[0])):
            set_norms()
            set_weights()
            if (day == num_days):
                set_end_prices(np.transpose(prev_prices[:,j]))
            else:
                set_end_prices_non_final(np.transpose(prev_prices[:,j]), [values[j], values[j+1]])
            values[j] = regress(prev_prices[:,j])
    return values[0]

def main():
    calculate()
    get_all_weights()

if __name__ == "__main__": main()
