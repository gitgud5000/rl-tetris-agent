import numpy as np
import tensorflow as tf
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from collections import deque
import random
import os

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, model_path=None, n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=10000)
        self.n_neurons = n_neurons
        self.activations = activations

        if model_path and os.path.isfile(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self._build_model()

        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0], kernel_initializer=HeUniform()))
        model.add(Dropout(0.2))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i], kernel_initializer=HeUniform()))
            model.add(Dropout(0.2))

        model.add(Dense(self.action_size, activation=self.activations[-1], kernel_initializer=HeUniform()))
        return model

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.model.predict(state, verbose=0)[0]

        action_probs = np.nan_to_num(action_probs, nan=0.0)
        action_probs = np.maximum(action_probs, 0)
        sum_probs = np.sum(action_probs)
        if sum_probs == 0:
            action_probs = np.ones(self.action_size) / self.action_size
        else:
            action_probs /= sum_probs

        action = np.random.choice(self.action_size, p=action_probs)
        return action, action_probs

    def train(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards = zip(*minibatch)
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        discounted_rewards = self._discount_rewards(rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards) > 0:
            discounted_rewards /= np.std(discounted_rewards)

        with tf.GradientTape() as tape:
            action_probs = self.model(states, training=True)
            action_masks = tf.one_hot(actions, self.action_size)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs + 1e-10), axis=1)
            entropy_loss = tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            loss = -tf.reduce_mean(log_probs * discounted_rewards) - 0.01 * entropy_loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.memory.clear()

    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            discounted_rewards[i] = cumulative
        return discounted_rewards

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
