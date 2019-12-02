from chemo_simul import ChemoMDP, QLearningAlgorithm, ChemoFeatureExtractor_wrap
from util import simulate, ValueIteration
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

trials = 20000
num_ranges = 100

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

mdp = ChemoMDP(max_months=6, a=.1, b=1.5, x=.15, y=1.2, d = 0.5, curedReward=5, deathReward=-5, k=50)

rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                            ChemoFeatureExtractor_wrap(50),
                                0.2)

learning_rewards = simulate(mdp, rl, trials, verbose=False)
rl.explorationProb = 0
optimal_pol_rewards = simulate(mdp, rl, trials, verbose=False)

wmin = 0
wmax = 20
mmin = 0
mmax = 15
#i is wellness, j is tumor
hmap = np.zeros((wmax-wmin,mmax-mmin))
for i in range(wmin,wmax):
	for j in range(mmin, mmax):
		state = (i / 10, j / 10, 1)
		hmap[i-wmin][j-mmin] = max((rl.getQ(state, action), action) for action in rl.actions(state))[1]

row_labels = np.arange(wmin, wmax)
col_labels = np.arange(mmin, mmax)

im, cbar = heatmap(hmap, row_labels, col_labels, cbarlabel="Dosage")
plt.xlabel('Tumor Size')
plt.ylabel('Wellness Measure')
plt.title('Learned Optimal Action at Patient States')
plt.show()