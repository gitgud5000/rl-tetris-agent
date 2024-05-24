## Overview
This project implements reinforcement learning agents that learn to play the classic game of Tetris. 
The first agent uses a Deep Q-Network (DQN) to learn a policy that maximizes the expected cumulative reward over time.
The second agent uses a Reinforce Policy Gradient method to learn a policy to increase the likelihood of taking actions that result in a higher return.

## Reinforce Policy Gradient Agent
* **Memory Deque** - Storing experiences (state, action, reward) with a maximum length of 10,000.
* **Optimizer** - Adam optimizer with a specified learning rate.
* **Neural Network** - Input layer, hidden layers, dropout layers, and output layer.
* **Experience Replay** - Storing and sampling experiences to stabilize training.
* **Discounted Rewards** - Valuing future rewards with a discount factor.
* **Reward Normalisation** - Stabilising training by normalising rewards.
* **Stochastic Gradient Descent** - Optimizing the policy network parameters.
* **Gradient Clipping** - Preventing gradient explosion.
* **Entropy Regularisation** - Encouraging exploration by adding entropy to the loss. 

![image](https://github.com/gitgud5000/rl-tetris-agent/assets/39888797/ad8e66cf-3c96-4e88-86d3-1f543c94681b)
