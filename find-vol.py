import matplotlib.pyplot as plt
import math
import numpy as np

all_prices = []
returns = []

def main():
    with open('data/real-asset-data/CO1.txt') as f:
        for line in f:
            all_prices.append(float(line))
    for i in range(len(all_prices)):
        if (i > 0):
            returns.append(math.log(float(all_prices[i]) / float(all_prices[i - 1])))

    avg_return = np.divide(sum(returns), len(returns))
    sq_dev = 0
    for i in range(len(returns)):
        sq_dev += pow((returns[i] - avg_return), 2)

    sq_dev = np.divide(sq_dev, len(returns))
    vol = math.sqrt(sq_dev)
    print vol * math.sqrt(252)

if __name__ == "__main__":
    main()
