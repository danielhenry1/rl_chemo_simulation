from submission import ChemoMDP


smallMDP = submission.ChemoMDP(cardValues=[1,5], multiplicity=2, threshold=10, peekCost=1)
mdp = smallMDP
mdp.computeStates()
rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                               submission.identityFeatureExtractor,
                               0.2)
util.simulate(mdp, rl, 30000)
print(rl.weights)