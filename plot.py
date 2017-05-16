import matplotlib.pyplot as plt

def main():
  plt.plotfile('prices.txt', delimiter=' ', cols=(0,1,2),
               names=('t','untraded', 'correlated'), marker='o')
  plt.show()
if __name__ == "__main__": main()
