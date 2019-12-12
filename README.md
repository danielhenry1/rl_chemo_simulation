# rl_chemo_simulation
Reinforcement Learning project for Stanford University CS221. Simulating the effects of chemotherapy on a patient and developing an optimal drug schedule.

We utilized the Reinforcement Learning template provided through the CS221 Coursework and implemented custom MDPs to simulate the effects of cancer and chemotherapy on the human body. We built our MDPs with guidance from existing literature and verified their results.

main.py will run out of the box and provide graphs showing the RL agent learning to optimize rewards in the two MDPs. In
both cases, the agent is only running for a short time, but the upward trand should be discernible.

Our MDP and Q-learning code can be found across chemo_simul.py, chemo_simul_complex.py and util.py

plot_vary_epsilon.py and heatmap.py include code that we used to generate some of our figures

Feel free to look through the full report "Final_Report" in this repository!