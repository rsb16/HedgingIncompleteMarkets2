import numpy as np
import matplotlib.pyplot as plt
import sys

num_assets 	= 2
current_day = int(sys.argv[1]) - 1
num_days 	= int(sys.argv[1])
data 		= []

for i in range(num_assets+1):
	for j in range(num_days):
		data.append(np.loadtxt("./asset"+str(i)+'.txt')[j::num_days])

while current_day > 0:

	asset_1 	= num_days + (current_day)
	asset_2 	= 2 * num_days + (current_day)
	sigma_day 	= np.cov(data[asset_1],data[asset_2])

	result = np.zeros(shape=(num_assets+1,num_assets+1))
	for x in range(len(sigma_day)):
	  for y in range(len(sigma_day[x])):
		  result[x][y] = 2*sigma_day[x][y]

	#add for loop for more than 2 assets
	result[0][num_assets] = -1 * np.mean(data[asset_1])
	result[1][num_assets] = -1 * np.mean(data[asset_2])
	result[num_assets][0] = -1 * np.mean(data[asset_1])
	result[num_assets][1] = -1 * np.mean(data[asset_2])

	cov1 = np.cov(data[current_day],data[asset_1])
	cov2 = np.cov(data[current_day],data[asset_2])
	temp = np.dot(np.linalg.inv(result),np.array([-2*cov1[0][1],-2*cov2[0][1],-np.mean(data[current_day])]))

	print 'Decisions: '
	print temp
	print 'Hedge value: ', (temp[0]*np.mean(data[asset_1]) + temp[1]*np.mean(data[asset_2]))
	print 'Stock value: ', + np.mean(data[current_day])

	current_day -= 1

# delta = 0.025
# x = np.arange(-10.0, 10.0, delta)
# y = np.arange(-10.0, 10.0, delta)
# X, Y = np.meshgrid(x, y)
# w = np.array([X,Y])
# C = np.array([cov1[0][1], cov2[0][1]])
# Z = np.ones((x.size, y.size))
# for i in range(x.size):
# 	for j in range(y.size):
# 		w = np.array([X[i,j], Y[i,j]])
# 		Z[i,j] = np.dot(np.transpose(w),np.dot(sigma,w) + map(times2,C))


# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
# plt.figure()
# CS = plt.contour(X, Y, Z)
# plt.annotate('Start point', xy=(temp[0],temp[1]), xytext=(0,0), textcoords='offset points')
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title('Value of cost function for weights')
# plt.show()

# contour labels can be placed manually by providing list of positions
# (in data coordinate). See ginput_manual_clabel.py for interactive
# placement.


















# print 'X'
# print X

# print 's1'
# print np.mean(setdata[1])
# print 's2'
# print np.mean(setdata[2])
