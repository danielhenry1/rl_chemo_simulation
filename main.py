from builtins import range, int, sum, len

from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractorWrapper
from chemo_simul_complex import ChemoMDPComplex, QLearningAlgorithmComplex, ChemoComplexFeatureExtractorWrapper
from util import simulate, ValueIteration
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from heatmap import heatmap

################################################

#SECTION 1: Simple Chemotherapy Models

################################################

### Part 1: Plotting Varying Epsilon ###

trials = 100000
num_ranges = 100
no_exp_trials = 20000
num_ranges_no_exp = 20
k = 20


# Function that creates a new QLearningAlgorithm depending on the epsilon decay, lets it run for a certain number of trials,
# then calculates the average rewards over a range
def learn(exp, decay, mdp, feature_extractor, trials, no_exp_trials):
	rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
								feature_extractor,
								exp, decay)
	print("Beginning {} trials with exploration")
	total_rewards, cured, died = simulate(mdp, rl, trials, verbose=False)
	rl.explorationProb = 0
	print("Beginning {} trials with no exploration")
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

mdp = ChemoMDP(max_months=6, a=.1, b=1.2, x=.15, y= .75, d = .5, curedReward=500, deathReward=-500, k=k)

# Initialize plot
xaxis = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 1)[1:]
xaxis_mean = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 10)[1:]
N = 10
plt.title('Rewards over time TS Model', fontsize = 20)


rl_rewards, rl_cured, rl_died = learn(0, 'Normal', mdp, ChemoFeatureExtractorWrapper(k), trials, no_exp_trials)
rolling_mean = pd.Series(rl_rewards).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean, label = 'k = {}'.format(k), linewidth=5.0)

plt.legend(loc='lower right')
plt.xlabel('Number of Trials', fontsize = 20)
plt.ylabel('Rolling Average Rewards', fontsize = 20)
plt.show()


# ############################################################

# # Section 2: Complex Chemotherapy Models

# ############################################################

trials = 70000
num_ranges = 80
no_exp_trials = 10000
num_ranges_no_exp = 20
k = 20

mdp = ChemoMDPComplex(n_cells_init=0.6, t_cells_init=0.4, i_cells_init=0.9)

# Initialize plot
xaxis = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 1)[1:]
xaxis_mean = np.linspace(0, trials + no_exp_trials, num_ranges + num_ranges_no_exp - 10)[1:]
N = 10
plt.title('Rewards over time TS Model', fontsize = 20)


rl_rewards, rl_cured, rl_died = learn(0, 'Normal', mdp, ChemoComplexFeatureExtractorWrapper(k), trials, no_exp_trials)
rolling_mean = pd.Series(rl_rewards).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean, label = 'k = {}'.format(k), linewidth=5.0)

plt.legend(loc='lower right')
plt.xlabel('Number of Trials', fontsize = 20)
plt.ylabel('Rolling Average Rewards', fontsize = 20)
plt.show()

