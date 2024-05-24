from keras.models import Sequential, save_model, load_model, clone_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import tensorflow as tf

class DQNAgent:
    """"
    Deep Q-Learning Agent
    """

    def __init__(
        self,
        state_size,
        mem_size=10000,
        gamma=0.95,
        epsilon=1,
        epsilon_min=0,
        epsilon_stop_episode=500,
        n_neurons=[32, 32],
        activations=["relu", "relu", "linear"],
        loss="mean_squared_error",
        optimizer="adam",
        replay_start_size=None,
        target_update_frequency=10,
    ):
        assert len(activations) == len(n_neurons) + 1

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        # self.loss = tf.keras.losses.Huber()
        self.optimizer = optimizer
        self.target_update_frequency = target_update_frequency
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model()
        self.target_model = clone_model(self.model)  # Initialize target network)
        self.target_model.set_weights(self.model.get_weights())
        self.update_target_counter = 0

    def _build_model(self):
        '''Build Neural Net for Deep-Q learning Model'''
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))
        # Hidden layers
        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))
        # Output layer
        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, current_state, next_state, reward, done):
        '''Stores the current state, the next state, the reward and if the game is over'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def best_state(self, states):
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=128, epochs=5):
        """Trains the agent"""
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:
            # Sample random minibatch of transitions
            minibatch = random.sample(self.memory, batch_size)

            # Get the expected score for the next states
            next_states = np.array([x[1] for x in minibatch])
            next_qs = [x[0] for x in self.target_model.predict(next_states)]

            x, y = [],[]
            # Build dataset to fit the model
            for i, (state, _, reward, done) in enumerate(minibatch):
                new_q = reward
                if not done:
                    new_q = reward + self.gamma * next_qs[i]

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.model.fit(
                np.array(x),
                np.array(y),
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
            )

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

            # Every 𝐶 steps, update 𝜽_2 = 𝜽_1
            self.update_target_counter += 1
            if self.update_target_counter % self.target_update_frequency == 0:
                self.update_target_network()
    
    def save_model(self, filepath):
        save_model(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")