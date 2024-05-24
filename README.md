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

### Hyperparameters

| Hyperparameter  | Description                                       | Value                 |
|-----------------|---------------------------------------------------|-----------------------|
| EPISODES        | Number of episodes to run for training            | 5000                  |
| MAX_STEPS       | Maximum steps per episode                         | None                  |
| GAMMA           | Discount factor for future rewards                | 0.95                  |
| render_every    | Frequency of rendering the environment            | 50                    |
| LOG_EVERY       | Frequency of logging training progress            | 50                    |
| N_NEURONS       | Number of neurons in each hidden layer            | [32, 32]              |
| TRAIN_EVERY     | Frequency of training the model                   | 1                     |
| render_delay    | Delay for rendering (in seconds)                  | None                  |
| ACTIVATIONS     | Activation functions for each layer               | ['relu', 'relu', 'linear'] |
| LEARNING_RATE   | Learning rate for the optimizer                   | 0.001                 |
| MODEL_SAVE_PATH | Path to the directory where the model will be saved| Path('models')        |
| BATCH_SIZE      | Number of samples per batch used during training  | 128                   |
