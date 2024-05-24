import random
from time import sleep
from tetris import Tetris
import cv2

# Create Tetris game instance
game = Tetris()

# Main game loop
while not game.game_over:
    # Get all possible next states
    next_states = game.get_next_states()

    # Randomly select a move
    move = random.choice(list(next_states.keys()))

    # Make the move
    x, rotation = move
    score, game_over = game.play(x, rotation, render=True, render_delay=0.1)

    # Print score and game over status
    print("Score:", game.get_game_score())
    if game_over:
        print("Game Over!")

    # Add delay for visualization
    sleep(0.5)

# Close OpenCV window after game over
cv2.destroyAllWindows()
