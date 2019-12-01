from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractor
from chemo_simul_complex import ChemoMDPComplex, QLearningAlgorithmComplex, ChemoComplexFeatureExtractor
from util import simulate, ValueIteration
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################################################

#SECTION 1: Simple Chemotherapy Models

################################################

### Without plotting varying epsilon
# trials = 60000
# num_ranges = 60
#
# mdp = ChemoMDP(max_months=6, a=.1, b=1.2, x=.15, y=1.2, d = .5, curedReward=500, deathReward=-500, k=50)
#
# # print("about to val iter")
# # stime = time.time()
# # vi = ValueIteration()
# # vi.solve(mdp, .001)
#
# # print(vi.pi)
#
# # d = {}
# # for k,v in vi.pi.items():
# # 	d[str(k)] = v
#
# # with open('policy.txt', 'w') as outfile:
# #     json.dump(d, outfile)
# # print(time.time() - stime)
#
# #mdp.computeStates()
# rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
#                                ChemoFeatureExtractor,
#                                0.9, 'Lin')
# print("STATS MODE: ")
# print("Number of Trials: " + str(trials))
# print("Wellness: " + str(mdp.wellness))
# print("Tumor_size: " + str(mdp.tumor_size))
#
# print("beginning simulation")
# total_rewards = simulate(mdp, rl, trials, verbose=False)
# print("simulation finished")
#
# rl.explorationProb = 0
#
# print("new simul")
# new_rewards = simulate(mdp, rl, trials, verbose=False)
# print("new simul donezo")
#
#
# # print(rl.weights)
# samples = []
# for i in range(num_ranges-1):
# 	range_size = trials / num_ranges
# 	sample = total_rewards[int(i * range_size): int((i + 1) * range_size)]
# 	samples.append(sum(sample)/len(sample))
# for sample in samples[:30]:
# 	print(sample)
# xaxis = np.linspace(0, trials, num_ranges)[1:]
# plt.plot(xaxis, samples, 'r')
# print("no explorationprob")
#
#
# samples = []
# for i in range(num_ranges-1):
# 	range_size = trials / num_ranges
# 	sample = new_rewards[int(i * range_size): int((i + 1) * range_size)]
# 	samples.append(sum(sample)/len(sample))
# for sample in samples[:30]:
# 	print(sample)
#
# plt.plot(xaxis, samples, 'b')
# plt.show()


########
### PLOTTING VARYING EPSILON (COMMENT OUT IF DON"T WANT SINCE IT TAKES A LONG TIME)
###########

trials = 60000
num_ranges = 60

mdp = ChemoMDP(max_months=6, a=.1, b=1.2, x=.15, y=1.2, d = .5, curedReward=500, deathReward=-500, k=50)
print("STATS MODE: ")
print("Number of Trials: " + str(trials))
print("Wellness: " + str(mdp.wellness))
print("Tumor_size: " + str(mdp.tumor_size))

# Initialize plot

xaxis = np.linspace(0, trials * 2, num_ranges * 2 - 1)[1:]
xaxis_mean = np.linspace(0, trials * 2, num_ranges * 2 - 10)[1:]
f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Sharing X axis')
def running_mean(x, N):
    sum = np.sum(np.insert(x, 0, 0))
    return (sum[N:] - sum[:-N]) / float(N)

# Varying RLs
rl_normal = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.2, 'Normal')
rl_linear = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.9, 'Lin')
rl_linear_new = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.9, 'LinNew')
rl_exp = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.9, 'Exp')
rl_delay = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.9, 'Delay')

# Normal (0.2 for X trials then 0 for X trials)
print("Starting Normal")
total_rewards_normal = simulate(mdp, rl_normal, trials, verbose=False)
rl_normal.explorationProb = 0
new_rewards_normal = simulate(mdp, rl_normal, trials, verbose=False)

samples = []
for i in range(num_ranges-1):
	range_size = trials / num_ranges
	sample = total_rewards_normal[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample)/len(sample))

for i in range(num_ranges-1):
	range_size = trials / num_ranges
	sample = new_rewards_normal[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample)/len(sample))

N = 10
rolling_mean = pd.Series(samples).rolling(window=N).mean().iloc[N-1:].values

axarr[0].plot(xaxis, samples, label = '0.2 and then 0')
axarr[1].plot(xaxis_mean, rolling_mean, label = '0.2 and then 0 rolling avg')


# Linear (-0.00001 decay)
print("Starting Linear")
total_rewards_linear = simulate(mdp, rl_linear, trials, verbose=False)
rl_linear.explorationProb = 0
new_rewards_linear = simulate(mdp, rl_linear, trials, verbose=False)

samples = []
for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = total_rewards_linear[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = new_rewards_linear[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

rolling_mean = pd.Series(samples).rolling(window=N).mean().iloc[N-1:].values
axarr[0].plot(xaxis, samples, label = 'Linear Decay in Epsilon (0.00001)')
axarr[1].plot(xaxis_mean, rolling_mean, label = 'Linear Rolling Avg (0.00001)')

# Linear New (-0.01 decay)
print("Starting Linear New")
total_rewards_linear_new = simulate(mdp, rl_linear_new, trials, verbose=False)
rl_linear_new.explorationProb = 0
new_rewards_linear_new = simulate(mdp, rl_linear_new, trials, verbose=False)

samples = []
for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = total_rewards_linear_new[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = new_rewards_linear_new[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

rolling_mean = pd.Series(samples).rolling(window=N).mean().iloc[N-1:].values
axarr[0].plot(xaxis, samples, label = 'Linear Decay in Epsilon (0.01)')
axarr[1].plot(xaxis_mean, rolling_mean, label = 'Linear Rolling Average (0.01)')

# Exponential (0.99999 decay rate)
print("Starting Exponential")
total_rewards_exp = simulate(mdp, rl_exp, trials, verbose=False)
rl_exp.explorationProb = 0
new_rewards_exp = simulate(mdp, rl_exp, trials, verbose=False)

samples = []
for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = total_rewards_exp[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

for i in range(num_ranges - 1):
	range_size = trials / num_ranges
	sample = new_rewards_exp[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample) / len(sample))

rolling_mean = pd.Series(samples).rolling(window=N).mean().iloc[N-1:].values
axarr[0].plot(xaxis, samples, label = 'Linear Decay in Epsilon (0.99999)')
axarr[1].plot(xaxis_mean, rolling_mean)

# Delayed Exponential

plt.show()



############################################################

# Section 2: Complex Chemotherapy Models

############################################################

# trials = 20000
# num_ranges = 10
#
# mdp = ChemoMDPComplex(n_cells_init=0.6, t_cells_init=.4, i_cells_init=0.9)
#
# rl = QLearningAlgorithmComplex(mdp.actions, mdp.discount(),
#                                ChemoComplexFeatureExtractor,
#                                0.2)
#
#
# print("Beginning Complex Simulation:")
# total_rewards = simulate(mdp, rl, trials, verbose=False)
# print("Finished Complex Simulation")
#
# rl.explorationProb = 0
#
# print("New Simulation")
# new_rewards = simulate(mdp, rl, trials, verbose=False)
# print("New Simulation donezo")
#
#
# average_rewards = []
# for i in range(num_ranges-1):
# 	range_size = trials / num_ranges
# 	sample = total_rewards[int(i * range_size): int((i + 1) * range_size)]
# 	average_rewards.append(sum(sample)/len(sample))
# for sample in average_rewards[:30]:
# 	print(sample)
# xaxis = np.linspace(0, trials, num_ranges)[1:]
#
#
# plt.plot(xaxis, average_rewards, 'r')
#
#
# print("no explorationprob")
#
#
# samples = []
# for i in range(num_ranges-1):
# 	range_size = trials / num_ranges
# 	sample = new_rewards[int(i * range_size): int((i + 1) * range_size)]
# 	samples.append(sum(sample)/len(sample))
# for sample in samples[:30]:
# 	print(sample)
#
# plt.plot(xaxis, average_rewards, 'b')
# plt.show()
#
