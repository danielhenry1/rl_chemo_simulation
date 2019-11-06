import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return ["go", "quit"]
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0:
            if action == "go":
                succ = (1, .001, 15)
                fail = (-1, .999, -1)
                return [succ, fail]
            if action == "quit":
                return[(-1, 1, 1)]
        return []
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.


    #HandCost, PeekedCard, Deck
    #newstate, prob, reward
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)

        def cash_out(handcost, index, deck, prob):
            return ((handcost, None, None), prob, handcost)

        def update_state_after_take(handcost, index, deck, prob):
            new_deck = list(deck)
            new_card_value = self.cardValues[index]
            new_deck[index] = new_deck[index] - 1
            new_handcost = handcost + new_card_value
            if new_handcost > self.threshold:
                return ((new_handcost, None, None), prob, 0)
            elif sum(new_deck) == 0:
                return cash_out(new_handcost, index, tuple(new_deck), prob)
            else:
                return ((new_handcost, None, tuple(new_deck)), prob, 0)

        def take_card(handcost, deck, top_index_known=None):
            next_states = []
            if top_index_known is not None:
                return [update_state_after_take(handcost, top_index_known, deck, 1)]
            else:
                options = []
                for index, val in enumerate(self.cardValues):
                    probability = deck[index] / sum(deck)
                    if probability > 0:
                        options.append(update_state_after_take(handcost, index, deck, probability))
                return options

        def peek_card(handcost, deck):
            options = []
            for index, val in enumerate(self.cardValues):
                new_deck = list(deck)
                prob = new_deck[index] / sum(new_deck)
                if prob > 0:
                    s = (handcost, index, tuple(new_deck))
                    options.append((s, prob, -self.peekCost))
            return options

        handcost = state[0]
        peek_index = state[1]
        deck = state[2]
        if deck is None:
            return []
        if action == "Quit":
            return [cash_out(handcost, peek_index, deck, 1)]

        if peek_index is None:
            if action == "Take":
                return take_card(handcost, deck, top_index_known=None)
            elif action == "Peek":
                return peek_card(handcost, deck)
        else:
            if action == "Take":
                return take_card(handcost, deck, top_index_known=peek_index)
            elif action == "Peek":
                return []
            else:
                print("Missed a case!")
                print(state, action)
                return []
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    my_blackjack = BlackjackMDP([5,21], 5 , 20, 1)
    return my_blackjack
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

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

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)


def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE

    def print_compares(rl, vi):
        diff = 0
        same = 0
        take_count = 0
        ocount = 0
        for k, v in vi.pi.items():
            rl_action = max((rl.getQ(k, action), action) for action in rl.actions(k))[1]
            if rl_action == v:
                same += 1
            else:
                diff += 1
            if rl_action == "Take":
                take_count += 1
            else:
                ocount += 1
        print(take_count)
        print(ocount)
        print("diff \% is {}".format(diff / (diff+same)))
        print("same \% is {}".format(same / (diff + same)))


    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, 0.2)
    if mdp.multiplicity == 3:
         rl.explorationProb = 0
    util.simulate(mdp, rl, 30000)

    vi = ValueIteration()
    vi.solve(mdp, .001)

    print_compares(rl,vi)



#END_CODE

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    options = []
    options.append(((total,action),1))
    if counts is not None:
        presences = []
        for i, count in enumerate(counts):
            key = "CardIndex:"+str(i)+",Action:"+action+",Count:"+str(count)
            options.append((key,1))
            presences.append(1 if count > 0 else 0)
        options.append((str(tuple(presences))+action, 1))
    return options
    # END_YOUR_CODE

#simulate_QL_over_MDP((smallMDP, largeMDP), blackjackFeatureExtractor)
############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    vi = ValueIteration()
    vi.solve(original_mdp, .001)

    fixed_rl = util.FixedRLAlgorithm(vi.pi)
    rewards = util.simulate(newThresholdMDP, fixed_rl, 10000)
    avg = sum(rewards) / len(rewards)
    print("Average reward on fixed value iter {}".format(avg))

    smart_rl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, 0.2)
    learning_rewards = util.simulate(modified_mdp, smart_rl, 10000)
    new_avg = sum(learning_rewards) / len(learning_rewards)
    print("Average reward on learning value iter {}".format(new_avg))

    # END_YOUR_CODE

