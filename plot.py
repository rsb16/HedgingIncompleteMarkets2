import matplotlib.pyplot as plt
import math
import numpy as np

denault = []
graaf = []
av_g = []
av_d = []

def main():
    for i in range(1, 50):
        with open('data/same-computation-time/result-data-' + str(i) + '.txt') as f:
            data = f.read().split('\n')
            denault.append((float(data[0])))
            graaf.append((float(data[1])))

    plt.plot(graaf, label='mcls')
    plt.plot(denault, label='rdv')
    for i in range(1, 50):
        av_g.append(sum(graaf)/len(graaf))
        av_d.append(sum(denault)/len(denault))
    plt.grid(True)
    plt.legend(prop={'size':12})
    plt.show()

if __name__ == "__main__":
    main()
