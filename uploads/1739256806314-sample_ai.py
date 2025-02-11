
#!/usr/bin/env python3
import argparse
import json
import random
import sys

def main():
    parser = argparse.ArgumentParser(description="Connect 4 Python AI")
    parser.add_argument('--board', required=True, help='Board state as a JSON string')
    parser.add_argument('--player', required=True, type=int, help='AI player number')
    args = parser.parse_args()

    try:
        board = json.loads(args.board)
    except json.JSONDecodeError:
        print("-1", flush=True)
        sys.exit(1)

    # Determine board dimensions
    num_rows = len(board)
    num_cols = len(board[0]) if num_rows > 0 else 0

    valid_columns = []
    # A column is valid if the top cell (highest row index) is empty (0)
    for col in range(num_cols):
        if board[num_rows - 1][col] == 0:
            valid_columns.append(col)

    if not valid_columns:
        # No valid moves available
        print("-1", flush=True)
        sys.exit(0)

    # Choose a random valid column
    chosen_column = random.choice(valid_columns)
    print(chosen_column, flush=True)

if __name__ == '__main__':
    main()
