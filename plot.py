import matplotlib.pyplot as plt
import math
import numpy as np

#graaf = []
all_prices = []
returns = []
#av_g = []
#av_d = []

def main():
    #for i in range(1, 13):
    with open('data/real-asset-data/CO1.txt') as f:
        #data = f.read().split('\n')
        #print data
        for line in f:
            all_prices.append(float(line))
            #graaf.append(float(data[1]))
    #plt.plot(graaf, label='mcls')
    #plt.plot(denault, label='oil-prices')
    #for i in range(1, 13):
        #av_g.append(sum(graaf)/len(graaf))
        #av_d.append(sum(denault)/len(denault))
    #plt.plot(av_g, label='av_mcls')
    #plt.plot(av_d, label='av_den')
    for i in range(len(all_prices)):
        # prices.append(float(all_prices[i]))
        if (i > 0):
            returns.append(math.log(float(all_prices[i]) / float(all_prices[i - 1])))

    avg_return = np.divide(sum(returns), len(returns))
    sq_dev = 0
    for i in range(len(returns)):
        sq_dev += pow((returns[i] - avg_return), 2)

    sq_dev = np.divide(sq_dev, len(returns))
    vol = math.sqrt(sq_dev)
    print vol * math.sqrt(252)
    # plt.grid(True)
    # plt.legend(prop={'size':12})
    # plt.show()

if __name__ == "__main__":
    main()
