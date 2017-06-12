import matplotlib.pyplot as plt

graaf = []
denault = []
av_g = []
av_d = []

def main():
    for i in range(1, 100):
        with open('result-data-' + str(i) + '.txt') as f:
            data = f.read().split('\n')
            denault.append(float(data[0]))
            graaf.append(float(data[1]))
    plt.plot(graaf, label='mcls')
    plt.plot(denault, label='den')
    for i in range(1, 100):
        av_g.append(sum(graaf)/len(graaf))
        av_d.append(sum(denault)/len(denault))
    plt.plot(av_g, label='av_mcls')
    plt.plot(av_d, label='av_den')
    plt.grid(True)
    plt.legend(prop={'size':12})
    plt.show()

if __name__ == "__main__":
    main()
