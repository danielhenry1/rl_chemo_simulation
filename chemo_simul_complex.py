import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np
import random


# This chemotherapy model is based on "Reinforcement learning-based control of drug dosing for cancer chemotherapy treatment"
'''
States:

I(t) = Number of immune cells  
N(t) = Number of normal cells
T(t) = Number of tumor cells
C(t) = Drug Concentration

Constants
a1 = Fractional immune cell kill rate
a2 = Fractional tumor cell kill rate
a3 = Fractional normal cell kill rate
b1 = Reciprocal carrying capacity of tumor cells
b2 = Reciprocal carrying capacity of normal cells
c1 = Immune Cell competition term (comp. between tumor and immune cells)
c2 = Tumor Cell competition term (comp. between tumor and immune cells)
c3 = Tumor Cell competition term (comp. between normal and tumor cells)
c4 = Normal Cell competition term (comp. between normal and tumor cells)
d1 = Immune cell death rate
d2 = Decay rate of injected drug
r1 = Per unit growth rate of tumor cells
r2 = Per unit growth rate of normal cells
s = Immune cell influx rate
alpha = Immune cell threshold rate
rho = Immune repsonse rate

'''
class ChemoMDPComplex(util.MDP):

    def __init__(self, n_cells_init, t_cells_init, i_cells_init, \
        a1=0.32, a2=0.38, a3=0.10, b1=0.2, b2=0.2, c1=0.5, c2=0.4, c3=0.585, c4=0.6, \
        d1=0.25, d2=1, r1=1.4, r2=1, s=0.37, alpha=0.35, rho=0.01, max_months = 7):

        self.n_cells_init = n_cells_init
        self.t_cells_init = t_cells_init
        self.i_cells_init = i_cells_init
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.d1 = d1
        self.d2 = d2
        self.r1 = r1
        self.r2 = r2
        self.s = s
        self.alpha = alpha
        self.rho = rho
        self.max_months = max_months



        print("a1: " + str(self.a1))
        print("a2: " + str(self.a2))
        print("a3: " + str(self.a3))
        print("b1: " + str(self.b1))
        print("b2: " + str(self.b2))
        print("c1: " + str(self.c1))
        print("c2: " + str(self.c2))
        print("c3: " + str(self.c3))
        print("c4: " + str(self.c4))
        print("d1: " + str(self.d1))
        print("d2: " + str(self.d2))
        print("r1: " + str(self.r1))
        print("r2: " + str(self.r2))
        print("s: " + str(self.s))
        print("alpha: " + str(self.alpha))
        print("rho: " + str(self.rho))



    # Return the start state.
    def startState(self):
        #(num of normal cells, num of tumor cells, num of immune cells, drug concentration, max_months)
        return (self.n_cells_init, self.t_cells_init, self.i_cells_init, 0, 0)

    # Return set of actions
    def actions(self, state):
        return (0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)


    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE 

        n_cells, t_cells, i_cells, d_c, t = state

        #end state
        if n_cells is None and t_cells is None and i_cells is None: return []

        #max months reached
        end_treatment_reward = 3 * t_cells
        if t == self.max_months: return [((None, None, None, None, t), 1, end_treatment_reward)]
        
        results = []

        #New number of normal cells
        delta_n_cells = self.r2 * n_cells * (1 - self.b2 * n_cells) - self.c4 * n_cells * t_cells - self.a3 * n_cells * d_c 

        #New number of tumor cells
        delta_t_cells = self.r1 * t_cells * (1 - self.b1 * t_cells) - self.c2 * i_cells * t_cells - self.c3 * t_cells * n_cells \
        - self.a2 * t_cells * d_c

        #New number of immune cells
        delta_i_cells = self.s + ((self.rho * i_cells * t_cells) / (self.alpha + t_cells)) - self.c1 * i_cells * t_cells - self.d1 * i_cells \
        - self.a1 * i_cells * d_c

        #New drug concentration
        delta_d_c = - self.d2 * d_c + action


        # cured!
        if (delta_t_cells + t_cells) <= 0: return [((None, None, None, None, t), 1, 5)]

        #next state
        nextState = (delta_n_cells + n_cells, delta_t_cells + t_cells, delta_i_cells + i_cells, delta_d_c + d_c, t + 1)


        #Reward based on proportional t-cell decrease
        currReward = 0
        if (delta_t_cells) < t_cells:
            currReward = (t_cells - (delta_t_cells)) / (t_cells)
        
        
        #hazard function
        newProbLiving = (i_cells + delta_i_cells) / (delta_t_cells  + t_cells)

        results.append((nextState,  min(newProbLiving, 1), currReward))

        #Death State
        deathState = (None, None, None, None, t+1)
        results.append((deathState, 1 -  min(newProbLiving, 1), -5))

        return results

        # END_YOUR_CODE

    def discount(self):
        return 1


def ChemoComplexFeatureExtractorWrapper(k):
    def ChemoComplexFeatureExtractor(state, action):
        n_cells, t_cells, i_cells, d_c, t = state
        features = []
        if n_cells is not None:
            n_bucket = math.floor(n_cells * k)
            features.append(("normal cells" + str(n_bucket) + str(action),1))
        if t_cells is not None:
            t_bucket = math.floor(t_cells * k)
            features.append(("tumor cells" + str(t_bucket) + str(action),1))
        if i_cells is not None:
            i_bucket = math.floor(i_cells * k)
            features.append(("immune cells" + str(i_bucket) + str(action),1))
        if n_cells is not None and t_cells is not None and i_cells is not None:
            features.append(("normal cells" + str(n_bucket) + "tumor cells" + str(t_bucket) + "immune cells" + str(i_bucket) + str(action),1))
        return features
    return ChemoComplexFeatureExtractor
    

############################################################
# Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithmComplex(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        qopt = self.getQ(state,action)
        vopt = 0
        if newState is not None:
            vopt = max([self.getQ(newState,next_action) for next_action in self.actions(newState)])
        scale = self.getStepSize() * (qopt - (reward + (self.discount * vopt)))
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - scale
        # END_YOUR_CODE
