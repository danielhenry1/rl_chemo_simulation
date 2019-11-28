import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


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
row = Immune repsonse rate

'''

class ChemoMDPComplex(util.MDP):
    def __init__(self, n_cells_init, t_cells_init, i_cells_init, \
        a1=0.2, a2=0.3, a3=0.1, b1=1, b2=1, c1=1, c2=0.5, c3=1, c4=1, \
        d1=0.2, d2=1, r1=1.5, r2=1, s=0.33, alpha=0.3, row=0.01):

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
        self.row = row


    # Return the start state.
    def startState(self):
        #(num of normal cells, num of tumor cells, num of immune cells, drug concentration)
        return (self.n_cells_init, self.t_cells_init, self.i_cells_init, 0)

    # Return set of actions possible from |state|.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return (0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.


    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE 

        n_cells, t_cells, i_cells, d_c = state

        #print("ACTIONS: " + str(action))
        #print("t_cells: " + str(t_cells))
        # Terminal state
        if n_cells is None and t_cells is None and i_cells is None: return []

        # # cured!
        # if M <= 0: return [((None, None, t), 1, self.curedReward)]

        #CALCULATE REWARD LATER

        end_treatment_reward = 10
        if t_cells < 0.05: return [((None, None, None, 0), 1, end_treatment_reward)]
        
        results = []

        #New number of normal cells
        new_n_cells = self.r2 * n_cells * (1 - self.b2 * n_cells) - self.c4 * n_cells * t_cells - self.a3 * n_cells * d_c 

        #New number of tumor cells
        new_t_cells = self.r1 * t_cells * (1 - self.b1 * t_cells) - self.c2 * i_cells * t_cells - self.c3 * t_cells * n_cells \
        - self.a2 * t_cells * d_c

        #New number of immune cells
        new_i_cells = self.s + ((self.row * i_cells * t_cells) / (self.alpha + t_cells)) - self.c1 * i_cells * t_cells - self.d1 * i_cells \
        - self.a1 * i_cells * d_c

        #New drug concentration
        new_d_c = - self.d2 * d_c + action

        #next state
        nextState = (new_n_cells, new_t_cells, new_i_cells, new_d_c)

        #Reward Function based on paper
        currReward = 0
        if new_t_cells < t_cells:
            currReward = (t_cells - new_t_cells) / (t_cells)

        #Living State
        if t_cells > 1:
            newProbLiving = 0
        else:
            newProbLiving = t_cells / 1
        results.append((nextState, newProbLiving, currReward))

        #Death State
        deathState = (None, None, None, 0)
        results.append((deathState, 1 - newProbLiving, currReward))

        # cured!


        if new_t_cells <= 0: return [((None, None, None, 0), 1, 20)]


        return results




        # END_YOUR_CODE

    def discount(self):
        return 1

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def ChemoComplexFeatureExtractor(state, action):
    n_cells, t_cells, i_cells, d_c = state
    features = []
    if n_cells is not None:
        n_bucket = n_cells * 10 // 1
        features.append(("normal cells" + str(n_bucket) + str(action),1))
    if t_cells is not None:
        t_bucket = t_cells * 10 // 1
        features.append(("tumor cells" + str(t_bucket) + str(action),1))
    if i_cells is not None:
        i_bucket = i_cells * 10 // 1
        features.append(("immune cells" + str(i_bucket) + str(action),1))
    if n_cells is not None and t_cells is not None and i_cells is not None:
        features.append(("normal cells" + str(n_bucket) + "tumor cells" + str(t_bucket) + "immune cells" + str(i_bucket) + str(action),1))
    return features

    

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
