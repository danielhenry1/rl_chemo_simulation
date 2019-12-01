from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractor
from chemo_simul_complex import ChemoMDPComplex, QLearningAlgorithmComplex, ChemoComplexFeatureExtractor
from util import simulate, ValueIteration
import time
import json
import matplotlib.pyplot as plt
import numpy as np

################################################

#SECTION 1: Simple Chemotherapy Models

################################################

trials = 60000
num_ranges = 60

mdp = ChemoMDP(max_months=6, a=.1, b=1.2, x=.15, y=1.2, d = .5, curedReward=500, deathReward=-500, k=50)

# print("about to val iter")
# stime = time.time()
# vi = ValueIteration()
# vi.solve(mdp, .001)

# print(vi.pi)

# d = {}
# for k,v in vi.pi.items():
# 	d[str(k)] = v

# with open('policy.txt', 'w') as outfile:
#     json.dump(d, outfile)
# print(time.time() - stime)

#mdp.computeStates()
rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                               ChemoFeatureExtractor,
                               0.9, 'Lin')
print("STATS MODE: ")
print("Number of Trials: " + str(trials))
print("Wellness: " + str(mdp.wellness))
print("Tumor_size: " + str(mdp.tumor_size))

print("beginning simulation")
total_rewards = simulate(mdp, rl, trials, verbose=False)
print("simulation finished")

rl.explorationProb = 0

print("new simul")
new_rewards = simulate(mdp, rl, trials, verbose=False)
print("new simul donezo")


# print(rl.weights)
samples = []
for i in range(num_ranges-1):
	range_size = trials / num_ranges
	sample = total_rewards[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample)/len(sample))
for sample in samples[:30]:
	print(sample)
xaxis = np.linspace(0, trials, num_ranges)[1:]
plt.plot(xaxis, samples, 'r')
print("no explorationprob")


samples = []
for i in range(num_ranges-1):
	range_size = trials / num_ranges
	sample = new_rewards[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample)/len(sample))
for sample in samples[:30]:
	print(sample)

plt.plot(xaxis, samples, 'b')
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
