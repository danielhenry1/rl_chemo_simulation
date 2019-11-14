from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractor
from util import simulate, ValueIteration
import time
import json

trials = 30000
num_ranges = 300

mdp = ChemoMDP(wellness=.2, tumor_size=.5, max_months=6, a=.1, b=1.2, x=.15, y=1.2, d = .5, curedReward=500, deathReward=-500)

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
                               0.2)
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

print("no explorationprob")


samples = []
for i in range(num_ranges-1):
	range_size = trials / num_ranges
	sample = new_rewards[int(i * range_size): int((i + 1) * range_size)]
	samples.append(sum(sample)/len(sample))
for sample in samples[:30]:
	print(sample)