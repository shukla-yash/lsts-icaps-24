# Source code for Specifications-guided-Dynamic-Task-Sampling-for-RL-Agents

Abstract: Reinforcement Learning (RL) has made significant strides in enabling artificial agents to learn diverse behaviors. However, learning an effective policy often requires a large number of environment interactions. To mitigate sample complexity issues, recent approaches have used high-level task specifications, such as Linear Temporal Logic (LTL_$) formulas or Reward Machines (RM), to guide the learning progress of the agent. In this work, we propose a novel approach, called Logical Specifications-guided Dynamic Task Sampling (LSTS), that learns a set of RL policies to guide an agent from an initial state to a goal state based on a high-level task specification, while minimizing the number of environmental interactions. Unlike previous work, LSTS does not assume information about the environment dynamics or the Reward Machine, and dynamically samples promising tasks that lead to successful goal policies. We evaluate LSTS on a gridworld and show that it achieves improved time-to-threshold performance on complex sequential decision-making problems compared to state-of-the-art RM and Automaton-guided RL baselines, such as Q-Learning for Reward Machines and Compositional RL from logical Specifications (DIRL). Moreover, we demonstrate that our method outperforms RM and Automaton-guided RL baselines in terms of sample-efficiency, both in a partially observable robotic task and in a continuous control robotic manipulation task.

Paper: https://arxiv.org/abs/2402.03678

----

To run the code, follow these steps:

Step 1: Install the official minigrid library:

`$ pip install minigrid`

Step 2: We need to update the minigrid repo with our version of minigrid that has the correct environments for our experiments:

`$ mv minigrid/ {path-to-minigrid-library}/`

Step 3: Head back to the LSTS repo:

`$ cd lsts-icaps-24/LSTS`

Step 4: Run the LSTS code

`$ python main.py`

Step 5: In order to plot the results and produce learning curves similar to fig 1 of the paper, do:

`$ python plot.py`

Important: Replace `log_dir` with path-to-logs. It is defined on `line 51` in `train.py`
