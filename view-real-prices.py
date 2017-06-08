# import matplotlib.pyplot as plt
import json


def main():
        json_data = json.load(open('flight-data-test.txt', 'r'))
        dict = {}
        for js in json_data:
            dict[str(js['origin'])] += js['price']

        print(dict)
            # print(json.loads(js))
    # plt.plotfile('untraded-prices.txt', delimiter=' ', cols=(0,1),
    #              names=('t','untraded'), marker='o', c='r')
    # plt.plotfile('correlated-prices.txt', delimiter=' ', cols=(0,1),
    #              names=('t','correlated'), marker='*', c='g')
    # plt.plotfile('profitshedging10x.txt', delimiter=',', cols=(0,1),
    #              names=('t','E(x)'), marker='o')
    # plt.show()
if __name__ == "__main__": main()