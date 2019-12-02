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

mdp = ChemoMDP(max_months=6, a=.1, b=4, x=.15, y= 1.2, d = .5, curedReward=500, deathReward=-500, k=50)
print("STATS MODE: ")
print("Number of Trials: " + str(trials))
print("Wellness: " + str(mdp.wellness))
print("Tumor_size: " + str(mdp.tumor_size))

###### VARYING EPSILON ######

trials = 60000
num_ranges = 60
no_exp_trials = 10000
num_ranges_no_exp = 100

def learn(exp, decay, mdp, trials, no_exp_trials):
	rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
								ChemoFeatureExtractor,
								exp, decay)
	total_rewards, cured, died = simulate(mdp, rl, trials, verbose=False)
	rl.explorationProb = 0
	new_rewards, cured_new, died_new = simulate(mdp, rl, no_exp_trials, verbose=False)

	samples = []
	for i in range(num_ranges - 1):
		range_size = trials / num_ranges
		sample = total_rewards[int(i * range_size): int((i + 1) * range_size)]
		samples.append(sum(sample) / len(sample))

	for i in range(num_ranges_no_exp - 1):
		range_size = no_exp_trials / num_ranges_no_exp
		sample = new_rewards[int(i * range_size): int((i + 1) * range_size)]
		samples.append(sum(sample) / len(sample))
	return samples, cured_new, died_new

# Initialize plot

xaxis = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 1)[1:]
xaxis_mean = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 10)[1:]
# f, axarr = plt.subplots(2, sharex=True)
# N = 10
# axarr[0].set_title('Average Reward of Every 500 Trials with Varying Exploration')
N = 10
# plt.title('Rolling Average Reward with Varying Exploration', fontsize = 20)
#

# # No Exploration
# print("Starting No Exploration")
# rewards, cured, died = learn(0, 'Normal', mdp, trials, no_exp_trials)
# rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
# # axarr[0].plot(xaxis, rewards, label = 'Constant Epsilon')
# # axarr[1].plot(xaxis_mean, rolling_mean, label = 'Constant Rolling Average')
# plt.plot(xaxis_mean, rolling_mean, label = 'No Exploration', linewidth=5.0)

# # Normal
# print("Starting Normal")
# rewards, cured, died = learn(0.2, 'Normal', mdp, trials, no_exp_trials)
# rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
# # axarr[0].plot(xaxis, rewards, label = 'Constant Epsilon')
# # axarr[1].plot(xaxis_mean, rolling_mean, label = 'Constant Rolling Average')
# plt.plot(xaxis_mean, rolling_mean, label = 'Constant Epsilon', linewidth=5.0)


# Linear
print("Starting Linear")
rewards_lin, cured_lin, died_lin = learn(0.9, 'Lin', mdp, trials, no_exp_trials)
rolling_mean_lin = pd.Series(rewards_lin).rolling(window=N).mean().iloc[N-1:].values
# axarr[0].plot(xaxis, rewards_lin, label = 'Linear Decay')
# axarr[1].plot(xaxis_mean, rolling_mean_lin, label = 'Linear Rolling Average')
# plt.plot(xaxis_mean, rolling_mean_lin, label = 'Linear Decay in Epsilon', linewidth=5.0)

# # Exponential
# print("Starting Exponential")
# rewards, cured = learn(0.9, 'Exp', mdp, trials, no_exp_trials)
# rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
# # axarr[0].plot(xaxis, rewards, label = 'Exponential Decay')
# # axarr[1].plot(xaxis_mean, rolling_mean, label = 'Exponential Rolling Average')
# plt.plot(xaxis_mean, rolling_mean, label = 'Exponential Decay in Epsilon', linewidth=5.0)
#
# print("plotting")
# # axarr[0].legend(loc='lower right')
# # axarr[1].legend(loc='lower right')
# plt.legend(loc='lower right')
# # Set common labels
# # axarr[1].set_xlabel('Number of Trials')
# # axarr[0].set_ylabel('Average Rewards')
# # axarr[1].set_ylabel('Average Rewards')
# plt.xlabel('Number of Trials', fontsize = 20)
# plt.ylabel('Rolling Average Rewards', fontsize = 20)
# plt.show()
# print("all done")

#### NAIVE ####
naive_trials = 1000

def naive(dose):
	def naive_actions():
		return dose
	rl_naive = QLearningAlgorithm(mdp.actions, mdp.discount(),
								  ChemoFeatureExtractor,
								  0.2, 'Normal',dose)
	total_rewards, cured, died = simulate(mdp, rl_naive, naive_trials, verbose=False)
	# samples = []
	# for i in range(num_ranges - 1):
	# 	range_size = trials / num_ranges
	# 	sample = total_rewards[int(i * range_size): int((i + 1) * range_size)]
	# 	samples.append(sum(sample) / len(sample))
	return np.mean(total_rewards), np.mean(cured), np.mean(died)



# Bar graph
objects = ('0.2', '0.5', '0.7', '1', 'RL')
y_pos = np.arange(len(objects))
doses = [0.2, 0.5, 0.7, 1]
percent_cured = []
percent_died = []
total_rewards = []
for dose in doses:
	rewards, cured, died = naive(dose)
	percent_cured.append(cured)
	percent_died.append(died)
	total_rewards.append(rewards)
percent_cured.append(np.mean(cured_lin))
percent_died.append(np.mean(died_lin))
#percent_cured = [naive(0.2),naive(0.5),naive(0.7),naive(1),np.mean(cured_lin)]
#percent_died = [naive(0.2),naive(0.5),naive(0.7),naive(1),np.mean(cured_lin)]

plt.bar(y_pos, percent_cured, .35)
plt.bar(y_pos + 0.35, percent_died, .35)
plt.xticks(y_pos, objects)
plt.ylabel('Percent')
plt.title('Comparison of Different Dosage Regimens for a Certain Patient')

plt.show()
plt.bar(y_pos, total_rewards, .35)
plt.xticks(y_pos, objects)
plt.ylabel('Average Rewards of 1000 Patients')
plt.title('Comparison of Different Dosage Regimens for a Certain Patient')



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
