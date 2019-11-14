import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np

class ChemoMDP(util.MDP):
    def __init__(self, wellness, tumor_size, max_months, a, b, x, y, d, curedReward, deathReward):
        """
        """
        self.wellness = wellness
        self.tumor_size = tumor_size
        self.max_months = max_months
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        self.d = d
        self.curedReward = curedReward
        self.deathReward = deathReward


    # Return the start state.
    def startState(self):
        #wellness, tumorsize, month
        return (self.wellness, self.tumor_size, 0)

    # Return set of actions possible from |state|.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return np.linspace(0,1,11)

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

        W, M, t = state

        # Terminal state
        if W is None: return []

        # # cured!
        # if M <= 0: return [((None, None, t), 1, self.curedReward)]

        #CALCULATE REWARD LATER

        end_treatment_reward = -30 * (W+M)
        if t == self.max_months: return [((None, None, t), 1, end_treatment_reward)]
        
        results = []

        #calculate next values
        deltaW = self.a * M + self.b * (action - self.d)
        deltaM = self.x * W - self.y * (action - self.d)

        newHealthyState = (W + deltaW, M + deltaM, t + 1)

        #Reward Function based on deltas
        currReward = 0
        if deltaW < -.5:
            currReward += -deltaW
        elif deltaW > .5:
            currReward -= -deltaW
        if deltaM < -.5:
            currReward += -deltaM
        elif deltaM > .5:
            currReward -= -deltaM

        
        # currReward = 0
        # if deltaW < -.5:
        #     currReward += 0.01
        # elif deltaW > .5:
        #     currReward -= 0.01
        # if deltaM < -.5:
        #     currReward += 0.01
        # elif deltaM > .5:
        #     currReward -= 0.01

        #Living State
        newProbLiving = np.exp(-(W+M)) + .15
        results.append((newHealthyState, newProbLiving, currReward))

        #Death State
        deathState = (None, None, t + 1)
        results.append((deathState, 1 - newProbLiving, self.tumor_size*(-5)))

        # cured!
        if M + deltaM <= 0: return [((None, None, t), 1, self.tumor_size*(5))]


        return results




        # END_YOUR_CODE

    def discount(self):
        return 1

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def ChemoFeatureExtractor(state, action):
    W, M, t = state
    features = []
    if W is not None:
        w_bucket = W * 10 // 1
        features.append(("W" + str(w_bucket) + str(action),1))
    if M is not None:
        m_bucket = M * 10 // 1
        features.append(("M" + str(m_bucket) + str(action),1))
    if W is not None and M is not None:
        features.append(("W" + str(w_bucket) + "M" + str(m_bucket) + str(action),1))
    return features

    

############################################################
# Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
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
