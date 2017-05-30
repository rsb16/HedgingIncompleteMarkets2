import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

num_assets 	= 2
current_day = 1
num_days 	= int(sys.argv[1])

asset_start_price = [250, 150, 80]
asset_mu = [0.07, 0.07, 0.12]
asset_sigma = [0.19, 0.13, 0.25]
norms_0 = np.zeros(num_days + 1)

DT = 1.0 / 365.0
INTEREST = 0
STRIKE = float(sys.argv[2])
RHO = [1, 0.66, 0.89]

current_portfolio_value = 0
trading_profit_day = 0

def set_norms():
	for i in range(len(norms_0)):
		norms_0[i] = np.random.randn()

def expected_next(asset_index):
	return asset_start_price[asset_index] * np.exp(asset_mu[asset_index] * DT)


def real(day, asset_index):
	if (asset_index == 0):
		return asset_start_price[0] * np.exp((asset_mu[0] - 0.5 * asset_sigma[0] * asset_sigma[0]) * DT + asset_sigma[0] * np.sqrt(DT) * norms_0[day])
	else:
		norm = RHO[asset_index] * norms_0[day] + np.sqrt(1 - RHO[asset_index]) * np.random.randn()
		return asset_start_price[asset_index] * np.exp((asset_mu[asset_index] - 0.5 * asset_sigma[asset_index] * asset_sigma[asset_index]) * DT + asset_sigma[asset_index] * np.sqrt(DT) * norm)

def covariance(current_day, i, j):
	return 		expected_next(i) \
				* expected_next(j) \
				* (np.exp(asset_sigma[i] * asset_sigma[j] * DT) - 1)

def w(t):
	bem = np.ones(num_assets)
	mus = np.ones(num_assets)
	rates = np.ones(num_assets)
	dt = (float(num_days) - float(current_day)) * DT
	for i in range(num_assets):
		bem[i] = (covariance(t, 0, i + 1) * (asset_sigma[0] / asset_sigma[i + 1]))
		mus[i] = asset_mu[i + 1]
		rates[i] = INTEREST * dt
	return np.dot(np.transpose((bem)), (mus - rates))


def d1(Xe, t):
	dt = (float(num_days) - float(current_day)) * DT
	return (np.log(float(Xe) / float(STRIKE)) + (w(t) + 0.5 * asset_sigma[0] * asset_sigma[0]) * (dt)) / (asset_sigma[0] * np.sqrt(dt))


def d2(Xe, t):
	dt = (float(num_days) - float(current_day)) * DT
	return (np.log(float(Xe) / float(STRIKE)) + (w(t) - 0.5 * asset_sigma[0] * asset_sigma[0]) * (dt)) / (asset_sigma[0] * np.sqrt(dt))

def V(Xe, t):
	#change delta_t to the proper value num_days/365.0
	dt = (float(num_days) - float(current_day)) * DT
	if (dt == 0):
		return max((Xe - STRIKE), 0)
	V = Xe * np.exp((w(t) - INTEREST)*(dt))*norm.cdf(d1(Xe,t)) - STRIKE * np.exp(-INTEREST * dt)*norm.cdf(d2(Xe,t))
	return V



# for i in range(num_assets+1):
# 	for j in range(num_days):
# 		data.append(np.loadtxt("./asset"+str(i)+'.txt')[j::num_days])


print '###########'
print 'START'
print '###########'
print 'untraded price : ' , asset_start_price[0]
print 'asset_1 price : ' , asset_start_price[1]
print 'asset_2 price : ' , asset_start_price[2]
print 'STRIKE : ', STRIKE
print 'Initial option price : ', V(asset_start_price[0], 0)
print 'Targetting : ', (V(asset_start_price[0], 0) - current_portfolio_value)

set_norms()
while current_day < num_days:

	result = np.ones(shape=(num_assets+1,num_assets+1))
	for x in range(len(asset_start_price) - 1):
	  for y in range(len(asset_start_price) - 1):
		  result[x][y] = 2 * covariance(current_day, x + 1, y + 1)


	#add for loop for more than 2 assets
	result[0][num_assets] = -1 * expected_next(1)
	result[1][num_assets] = -1 * expected_next(2)
	result[num_assets][0] = -1 * expected_next(1)
	result[num_assets][1] = -1 * expected_next(2)
	result[2][2] = 0

	cov1 = covariance(current_day, 1, 0)
	cov2 = covariance(current_day, 2, 0)
	a = V(expected_next(0), current_day + 1)
	# print 'Target: ', (max(V(expected_next(0), current_day + 1), 0) - current_portfolio_value)
	temp = np.dot(np.linalg.inv(result),np.array([-2*cov1,-2*cov2, -1 * (max(V(expected_next(0), current_day + 1), 0) - current_portfolio_value)]))#-1 * expected(current_day, 0)]))

	prev_0 = asset_start_price[0]
	prev_1 = asset_start_price[1]
	prev_2 = asset_start_price[2]
	real_0 = real(current_day, 0)
	real_1 = real(current_day, 1)
	real_2 = real(current_day, 2)
	print temp
	print 'Expected hedge value : ' , (temp[0] * expected_next(1) + temp[1] * expected_next(2))
	print '###########'
	print 'DAY : ' + str(current_day)
	print '###########'
	# print 'Decisions at day: ' , (current_day - 1)
	# print temp
	# print 'Expected Hedge value: ' + str((temp[0] * expected_next(1) + temp[1] * expected_next(2)))
	# print 'Real Hedge value: ' + str(temp[0] * real_1 + temp[1] * real_2 + trading_profit)
	current_portfolio_value += temp[0] * real_1 + temp[1] * real_2 + trading_profit_day
	# print 'Targetting: ' + str(V(asset_start_price[0], current_day + 1) - (current_portfolio_value))
	# print 'Expected Option value ' + str(current_day + 1) + ' : ' + str(V(asset_start_price[0], current_day + 1))
	# print 'Real Stock value: ' + str(real_0)
	# print 'Real Asset_1 value: ' + str(real_1)
	# print 'Real Asset_2 value: ' + str(real_2)
	trading_profit_day = temp[0] * (real_1 - prev_1) + temp[1] * (real_2 - prev_2)
	print 'TRADING_PROFIT: ', trading_profit_day
	print 'current_total_portfolio_value : ', current_portfolio_value

	asset_start_price[0] = real_0
	asset_start_price[1] = real_1
	asset_start_price[2] = real_2

	current_day += 1
	#
	# print "expected given: " + str(asset_start_price[0]) + " : " + str(expected_next(0))
	# print "expected given: " + str(asset_start_price[1]) + " : " + str(expected_next(1))
	# print "expected given: " + str(asset_start_price[2]) + " : " + str(expected_next(2))


# real_0 = real(current_day, 0)
# real_1 = real(current_day, 1)
# real_2 = real(current_day, 2)
# asset_start_price[0] = real_0
# asset_start_price[1] = real_1
# asset_start_price[2] = real_2


print '###########'
print 'END'
print '###########'
# print 'untraded price : ' , asset_start_price[0]
# print 'asset_1 price : ' , asset_start_price[1]
# print 'asset_2 price : ' , asset_start_price[2]
print 'Hedge value : ' , current_portfolio_value
print 'Payoff : ', V(asset_start_price[0], current_day)

# print (V(asset_start_price[0], current_day) - ((temp[0] * asset_start_price[1] + temp[1] * asset_start_price[2]) + trading_profit))

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
