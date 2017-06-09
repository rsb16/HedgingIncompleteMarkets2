import numpy as np
from scipy import optimize
import random
import itertools
INTEREST = 1.015
STRIKE = 50
DT = 1.0
num_days = 1
num_trials = 1000000
start_prices = [50, 60, 10, 10]
vols = [0.2, 0.3, 0.44, 0.11]
rhos = [1, 0.69, 0.22, 0.45]
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
            weights[i][j] = random.randint(-5, 10)

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
    terms = np.zeros(len(start_prices) - 1)
    for i in range(len(terms)):
        terms[i] = x[len(x) - len(start_prices) + 1 + i]
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
            A[i][- 1 * (len(higher_order_weights_diag)) + j] = higher_order_weights_diag[j]

    def loss(x):
        #print 'x : ', x
        #print 'H: ', H
        #print 'c: ', c
        return abs(0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

    x = np.dot(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(A), A), EPS)), np.transpose(A)), b)
    #print 'x: ', x
    c0 = x[0] # get the constant

    # get the single order terms
    single_order_terms = np.zeros(len(start_prices) - 1)

    for i in range(len(single_order_terms)):
        single_order_terms[i] = x[i + 1]

    second_order_terms_diag = get_second_order_terms_diag(x)

    second_order_terms = get_second_order_terms(x)

    H = np.zeros(shape=(len(start_prices) - 1, len(start_prices) - 1))

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
    x0 = np.ones(len(start_prices) - 1)
    res_uncons = optimize.minimize(loss, x0, method='Nelder-Mead', options=opt)
    value = 0
    for i in range(len(res_uncons.x)):
        value += res_uncons.x[i] * start_prices[i + 1]
    print value
    return value

for day in range(num_days):
    set_norms()
    set_weights()
    set_end_prices()
    regress()
