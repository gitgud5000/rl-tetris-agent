import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from dqn_agent import DQNAgent
from logs import CustomTensorBoard
from tetris import Tetris


def log_scores(log, episode, scores, log_every):
    """Log the average, minimum, and maximum scores."""
    min_score = min(scores[-log_every:])
    avg_score = mean(scores[-log_every:])
    max_score = max(scores[-log_every:])

    log.log(episode, avg_score=avg_score, min_score=min_score, max_score=max_score)


def initialize_environment(state_size, n_neurons, activations, epsilon_stop_episode, mem_size, gamma, replay_start_size, target_update_frequency):
    """Initialize the Tetris environment and agent."""
    env = Tetris()
    agent = DQNAgent(
        state_size,
        n_neurons=n_neurons, activations=activations,
        epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
        gamma=gamma, replay_start_size=replay_start_size, target_update_frequency=target_update_frequency
    )
    log_dir = f'logs/tetris-nn={n_neurons}-mem={mem_size}-bs={512}-e={1}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)
    return env, agent, log

def play_episode(env, agent, render_every, episode, render_delay, max_steps):
    """Play one episode of the game."""
    current_state = env.reset()
    done = False
    steps = 0
    render = render_every and episode % render_every == 0
    render = False

    # Episodic gameplay
    while not done and (not max_steps or steps < max_steps):
        next_states = env.get_next_states()
        best_state = agent.best_state(next_states.values())

        best_action = next(
            (action for action, state in next_states.items() if state == best_state),
            None,
        )

        reward, done = env.play(
            best_action[0], best_action[1], render=render, render_delay=render_delay
        )
        # Save the current state, next state, reward, and done status
        agent.remember(current_state, next_states[best_action], reward, done)
        current_state = next_states[best_action]
        steps += 1

    return env.get_game_score()

def save_agent(agent, path, episode,flg_date=True):
    """Save the agent's model and state."""
    if flg_date:
        today = datetime.today().strftime('%Y%m%d_%H_%M_%S')
        agent.save_model(os.path.join(path, f'{today}_tetris_dqn_model_{episode}.h5'))
    else:
        agent.save_model(os.path.join(path, f'tetris_dqn_model_{episode}.h5'))

def dqn():
    """Deep Q Learning training function for Tetris."""
    
    env = Tetris()
    EPISODES = 3005
    MAX_STEPS = None
    EPSILON_STOP_EPISODE = 1500
    MEM_SIZE = 20000
    GAMMA = 0.95
    BATCH_SIZE = 512
    EPOCHS = 1
    render_every = 50
    LOG_EVERY = 50
    replay_start_size = 2000
    TRAIN_EVERY = 1
    N_NEURONS = [32, 32]
    render_delay = None
    ACTIVATIONS = ['relu', 'relu', 'linear']
    TARGET_UPDATE_FREQUENCY = 10
    MODEL_SAVE_PATH = Path('models')
    MODEL_SAVE_PATH.mkdir(exist_ok=True)

    env, agent, log = initialize_environment(
        state_size=env.get_state_size(), n_neurons=N_NEURONS, activations=ACTIVATIONS,
        epsilon_stop_episode=EPSILON_STOP_EPISODE, mem_size=MEM_SIZE, gamma=GAMMA,
        replay_start_size=replay_start_size, target_update_frequency=TARGET_UPDATE_FREQUENCY
    )

    scores = []
    try:
        for episode in tqdm(range(EPISODES)):
            score = play_episode(env, agent, render_every, episode, render_delay, MAX_STEPS)
            scores.append(score)

            # train the agent with the experience of the episode
            if episode % TRAIN_EVERY == 0:
                agent.train(batch_size=BATCH_SIZE, epochs=EPOCHS)

            # Loggings
            if LOG_EVERY and episode and episode % LOG_EVERY == 0:
                log_scores(log, episode, scores, LOG_EVERY)
            
    # Save the final trained model
        save_agent(agent, MODEL_SAVE_PATH, 'complete')

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        save_agent(agent, MODEL_SAVE_PATH, 'interrupted')
    finally:
        save_agent(agent, MODEL_SAVE_PATH, 'last',flg_date=False)

# ----------------------------------------
# Play Tetris with the trained agent
# ----------------------------------------

def initialize_agent_for_play(model_path):
    """Initialize the Tetris environment and load the trained agent."""
    
    EPSILON= 0.01
    
    env = Tetris()
    agent = DQNAgent(env.get_state_size(),
                     epsilon=EPSILON,)
                     
    agent.load_model(model_path)
    return env, agent

def get_best_action(agent, next_states):
    """Get the best action from the available next states."""
    best_state = agent.best_state(next_states.values())
    best_action = next((action for action, state in next_states.items() if state == best_state), None)
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
    MODEL_PATH = "models/tetris_dqn_model_last.h5"
    env, agent = initialize_agent_for_play(MODEL_PATH)
    play_game(env, agent)


if __name__ == "__main__":
    # PARAMETER RUNTIME
    parser = ArgumentParser()
    parser.add_argument(
        f"--mode",
        type=str,
        choices=["train", "play"],
        required=False,
        default="train",
        # default='play'
    )

    args = parser.parse_args()
    if args.mode == "train":
        print("Training Tetris with DQN...")
        dqn()
    else:
        print("Playing Tetris with the trained agent...")
        play_tetris_with_trained_agent()