
#!/usr/bin/env python3
import argparse
import json
import numpy as np
import random

def get_valid_moves(board):
    """Return list of column indices where the top row is 0 (i.e. not full)."""
    return [col for col in range(board.shape[1]) if board[0, col] == 0]

def main():
    parser = argparse.ArgumentParser(description="Sample AI Agent for Connect 4")
    parser.add_argument('--board', required=True, type=str,
                        help='JSON-encoded board (a 2D list)')
    parser.add_argument('--player', required=True, type=int,
                        help='Player number (1 or -1)')
    args = parser.parse_args()

    # Convert JSON board to a numpy array
    board_list = json.loads(args.board)
    board = np.array(board_list)
    
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        print(-1)  # No valid moves available
    else:
        move = random.choice(valid_moves)
        print(move)

if __name__ == '__main__':
    main()

