import numpy as np
from scipy import optimize
import random
import itertools
INTEREST = 1.015
STRIKE = 50
DT = 1.0
num_days = 1
num_trials = 10
start_prices = [50, 60, 1]
vols = [0.2, 0.3, 0.44]
rhos = [1, 0.45, 0.22]
weights = np.zeros(shape=(len(start_prices) - 1, num_trials))
norms = np.zeros(shape=(num_trials, len(start_prices)))
end_prices = np.zeros(shape=(num_trials, len(start_prices)))
EPSILON = 1e-6

def set_norms():
    for i in range(len(norms)):
        for j in range(len(norms[0])):
            if (j == 0):
                norms[i][j] = np.random.randn()
            else:
                norms[i][j] = rhos[j] * norms[i][0] + np.sqrt(1 - rhos[j]) * np.random.randn()


def set_end_prices():
    for i in range(len(end_prices)):
        for j in range(len(end_prices[0]) - 1):
            if (norms[i][j] > 0):
                end_prices[i][j] = start_prices[j] * np.exp(vols[j] * np.sqrt(DT))
            else:
                end_prices[i][j] = start_prices[j] * np.exp(-1 * vols[j] * np.sqrt(DT))
        end_prices[i][0] = max(end_prices[i][0] - STRIKE, 0)
        end_prices[i][len(end_prices[0]) - 1] = start_prices[len(end_prices[0]) - 1] * np.exp((INTEREST - 1) * DT)

def set_weights():
    for i in range(len(start_prices) - 1):
        for j in range(num_trials):
            weights[i][j] = random.randint(-10, 10)

def triangle(n):
    return n * (n + 1) / 2

def get_diag(matrix):
    diag = np.zeros(len(matrix))
    for i in range(len(matrix)):
        diag[i] = matrix[i][i]
    return diag

def get_upper_triang(matrix):
    ut_indices = np.triu_indices(len(matrix))
    values = np.zeros(triangle(len(matrix) - 1))
    for i in range(len(values)):
        #probably need to test this
        values[i] = matrix[ut_indices][i + 1]
    return values

def get_second_order_terms(x):
    terms = np.zeros(len(x) - len(start_prices))
    for i in range(len(terms)):
        terms[i] = x[len(start_prices) + i]
    return terms

def symmetrize(a):
    return a + a.T - numpy.diag(a.diagonal())

def regress():
    b = np.zeros(num_trials)
    for i in range(len(b)):
        port = 0
        for j in range(len(start_prices) - 1):
            port += weights[j][i] * end_prices[i][j + 1]
        b[i] = end_prices[i][0] - port
    A = np.zeros(shape=(num_trials, triangle(len(start_prices))))
    EPS = np.zeros(shape=(triangle(len(start_prices)), triangle(len(start_prices))))

    for i in range(len(EPS)):
        for j in range(len(EPS[0])):
            EPS[i][j] = EPSILON

    for i in range(len(A)):
        A[i][0] = 1
        weights_trial = np.zeros(len(start_prices) - 1)
        upper_triangular_indices = np.triu_indices(len(start_prices) - 2)
        for j in range(len(start_prices) - 1):
            weights_trial[j] = weights[j][i]
            A[i][j + 1] = weights[j][i]
        shape = (len(start_prices) - 1, len(start_prices) - 1)
        higher_order_weights = (np.triu(np.array([x * y for (x,y) in list(itertools.product(weights_trial, weights_trial))]).reshape(shape)))
        higher_order_weights_diag = get_diag(np.array(np.diag(np.diag(higher_order_weights))))
        higher_order_weights_UT = get_upper_triang(np.array(np.subtract(higher_order_weights, np.diag(np.diag(higher_order_weights)))))
        for j in range(len(higher_order_weights_UT)):
            A[i][len(start_prices) + j] = higher_order_weights_UT[j]
        for j in range(len(higher_order_weights_diag)):
            A[i][(len(start_prices) - 1 + len(upper_triangular_indices)) + j] = higher_order_weights_diag[j]

    def loss(x):
        return abs(0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

    x = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(A), A), EPS)), np.transpose(A)), b)
    c0 = x[0] # get the constant

    # get the single order terms
    single_order_terms = np.zeros(len(start_prices) - 1)

    for i in range(len(single_order_terms)):
        single_order_terms[i] = x[i + 1]

    second_order_terms = get_second_order_terms(x)
    print second_order_terms
    H = np.zeros(shape=(len(start_prices) - 1, len(start_prices) - 1))

    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j]
    H = np.array([[second_order_terms[1],second_order_terms[0]],[second_order_terms[0], second_order_terms[2]]])
    print H
    c = single_order_terms
    print c

    opt = {'disp':False, 'maxiter':500}
    x0 = [1,1]
    res_uncons = optimize.minimize(loss, x0, method='Nelder-Mead',
                                       options=opt)

    print (res_uncons)

set_norms()
set_weights()
set_end_prices()
regress()
