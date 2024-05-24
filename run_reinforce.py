import os
from pathlib import Path
import random
from argparse import ArgumentParser
from datetime import datetime
from statistics import mean, median
import numpy as np
from tqdm import tqdm

from reinforce import PolicyGradientAgent
from logs import CustomTensorBoard
from tetris import Tetris


def log_scores(log, episode, scores, log_every):
    """Log the average, minimum, and maximum scores."""
    avg_score = mean(scores[-log_every:])
    min_score = min(scores[-log_every:])
    max_score = max(scores[-log_every:])

    log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)


def initialize_environment(state_size, action_size, n_neurons, activations, learning_rate, gamma):
    """Initialize the Tetris environment and agent."""
    env = Tetris()
    agent = PolicyGradientAgent(
        state_size, action_size, n_neurons=n_neurons, activations=activations,
        learning_rate=learning_rate, gamma=gamma
    )
    log_dir = f'logs/tetris-nn={n_neurons}-lr={learning_rate}-gamma={gamma}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)
    return env, agent, log


def play_episode(env, agent, render_every, episode, render_delay, max_steps):
    current_state = env.reset()
    current_state = np.reshape(current_state, [1, env.get_state_size()])
    done = False
    steps = 0
    render = render_every and episode % render_every == 0
    total_reward = 0

    while not done and (not max_steps or steps < max_steps):
        next_states = env.get_next_states()
        possible_actions = list(next_states.keys())

        action, action_probs = agent.act(current_state)
        action = min(action, len(possible_actions) - 1)
        x, rotation = possible_actions[action]

        reward, done = env.play(x, rotation, render=render, render_delay=render_delay)
        next_state = env._get_board_props(env.board)
        next_state = np.reshape(next_state, [1, env.get_state_size()])

        agent.remember(current_state, action, reward)

        current_state = next_state
        total_reward += reward
        steps += 1

    return total_reward



def save_agent(agent, path, episode):
    """Save the agent's model and state."""
    today = datetime.today().strftime('%Y%m%d_%H_%M_%S')
    agent.save_model(os.path.join(path, f'{today}_tetris_policy_gradient_model_{episode}.h5'))


def train():
    """Policy Gradient training function for Tetris."""

    env = Tetris()
    EPISODES = 2000
    MAX_STEPS = None
    GAMMA = 0.95
    render_every = 50
    LOG_EVERY = 50
    N_NEURONS = [32, 32]
    TRAIN_EVERY = 1
    render_delay = None
    ACTIVATIONS = ['relu', 'relu', 'linear']
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = Path('models')
    MODEL_SAVE_PATH.mkdir(exist_ok=True)

    env, agent, log = initialize_environment(
        state_size=env.get_state_size(), action_size=len(env.get_next_states()),
        n_neurons=N_NEURONS, activations=ACTIVATIONS, learning_rate=LEARNING_RATE, gamma=GAMMA
    )

    scores = []
    try:
        for episode in tqdm(range(EPISODES)):
            score = play_episode(env, agent, render_every, episode, render_delay, MAX_STEPS)
            scores.append(score)

            # train the agent with the experience of the episode
            if episode % TRAIN_EVERY == 0:
                agent.train()

            # Loggings
            if LOG_EVERY and episode and episode % LOG_EVERY == 0:
                log_scores(log, episode, scores, LOG_EVERY)

        # Save the final trained model
        save_agent(agent, MODEL_SAVE_PATH, 'final')

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        save_agent(agent, MODEL_SAVE_PATH, 'interrupted')


# ----------------------------------------
# Play Tetris with the trained agent
# ----------------------------------------

def initialize_agent_for_play(model_path):
    """Initialize the Tetris environment and load the trained agent."""
    env = Tetris()
    state_size = env.get_state_size()
    action_size = len(env.get_next_states())
    agent = PolicyGradientAgent(state_size, action_size)
    agent.load_model(model_path)
    return env, agent


def get_best_action(agent, next_states):
    """Get the best action from the available next states."""
    state = np.array(list(next_states.values()))
    action, _ = agent.act(state)
    best_action = list(next_states.keys())[action]
    return best_action


def play_game(env, agent):
    """Play a single game of Tetris with the trained agent."""
    current_state = env.reset()
    done = False

    while not done:
        next_states = env.get_next_states()
        best_action = get_best_action(agent, next_states)
        reward, done = env.play(best_action[0], best_action[1], render=True)
        current_state = next_states[best_action]


def play_tetris_with_trained_agent():
    """Play Tetris with a trained agent."""
    MODEL_PATH = 'tetris_policy_gradient_model.h5'
    env, agent = initialize_agent_for_play(MODEL_PATH)
    play_game(env, agent)


if __name__ == "__main__":

    # PARAMETER RUNTIME
    parser = ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        choices=[
            'train',
            'play'
        ],
        required=False,
        default='train'
    )

    args = parser.parse_args()
    if args.mode == 'train':
        print("Training Tetris with Policy Gradient...")
        train()
    else:
        print("Playing Tetris with the trained agent...")
        play_tetris_with_trained_agent()
