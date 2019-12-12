from builtins import range, int, sum, len

from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractorWrapper
from chemo_simul_complex import ChemoMDPComplex, QLearningAlgorithmComplex, ChemoComplexFeatureExtractorWrapper
from util import simulate, ValueIteration
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################################################

#SECTION 1: Simple Chemotherapy Models

################################################

### Part 1: Plotting Varying Epsilon ###

trials = 900
num_ranges = 6
no_exp_trials = 100
num_ranges_no_exp = 10

mdp = ChemoMDP(max_months=6, a=.1, b=1.2, x=.15, y= 1.2, d = .5, curedReward=500, deathReward=-500, k=50)
print("VARYING EPSILON: ")

ChemoFeatureExtractor = ChemoFeatureExtractorWrapper(10)

# Function that creates a new QLearningAlgorithm depending on the epsilon decay, lets it run for a certain number of trials,
# then calculates the average rewards over a range
def learn(exp, decay, mdp, feature_extractor, trials, no_exp_trials):
	rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
								feature_extractor,
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
N = 10
plt.title('Rolling Average Reward with Varying Exploration', fontsize = 20)

# No Exploration
print("Starting No Exploration")
rewards, cured, died = learn(0, 'Normal', mdp, ChemoFeatureExtractor, trials, no_exp_trials)
rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean, label = 'No Exploration', linewidth=5.0)

# Normal
print("Starting Normal")
rewards, cured, died = learn(0.2, 'Normal', mdp, ChemoFeatureExtractor, trials, no_exp_trials)
rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean, label = 'Constant Epsilon', linewidth=5.0)

# Linear
print("Starting Linear")
rewards_lin, cured_lin, died_lin = learn(0.9, 'Lin', mdp, ChemoFeatureExtractor, trials, no_exp_trials)
rolling_mean_lin = pd.Series(rewards_lin).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean_lin, label = 'Linear Decay in Epsilon', linewidth=5.0)

# Exponential
print("Starting Exponential")
rewards, cured, died = learn(0.9, 'Exp', mdp, ChemoFeatureExtractor, trials, no_exp_trials)
rolling_mean = pd.Series(rewards).rolling(window=N).mean().iloc[N-1:].values
plt.plot(xaxis_mean, rolling_mean, label = 'Exponential Decay in Epsilon', linewidth=5.0)

print("plotting")
plt.legend(loc='lower right')
plt.xlabel('Number of Trials', fontsize = 20)
plt.ylabel('Rolling Average Rewards', fontsize = 20)
plt.show()
print("all done")