
#!/usr/bin/env python
import argparse
import json
import logging
import math
import numpy as np
import time
from timeit import default_timer as timer
from scipy.signal import convolve2d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

###############################
# Helper Functions
###############################
class HelperFunctions:
    @staticmethod
    def place(choice: int, board, player: int):
        """Return a new board with the token dropped into the specified column.
           Assumes board index 0 is the top row.
        """
        board = board.copy()
        if board[0, choice] != 0:
            logging.info(f"Invalid move! Column {choice} is full.")
            return board
        row = board.shape[0] - 1
        while board[row, choice] != 0:
            row -= 1
        board[row, choice] = player
        return board

    @staticmethod
    def get_valid_moves(board):
        """Return a list of column indices where the top cell is 0 (i.e. not full)."""
        return np.where(board[0] == 0)[0].tolist()

    @staticmethod
    def check_win(board):
        """Uses convolution to detect a win. Returns:
           1 if player 1 wins,
          -1 if player -1 wins,
           0 if draw,
           None if the game is not over.
        """
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = horizontal_kernel.T
        diag1_kernel = np.eye(4, dtype=np.int32)
        diag2_kernel = np.fliplr(diag1_kernel)
        kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        for kernel in kernels:
            conv = convolve2d(board, kernel, mode='valid')
            if (conv == 4).any():
                return 1
            if (conv == -4).any():
                return -1

        if len(HelperFunctions.get_valid_moves(board)) == 0:
            return 0
        return None

###############################
# OmegaZero Agent
###############################
class OmegaZero:
    def __init__(self):
        self.transposition_table = {}
        self.TIME_LIMIT = 295  # Total game time (seconds)
        self.start_time = None
        self.max_depth = 1
        self.player = None
        self.time_per_move = 10  # Initial allocation per move (seconds)
        self.total_moves = 0
        self.previous_board = None

    def reset(self):
        """Reset internal state for a new game."""
        self.transposition_table = {}
        self.start_time = time.time()
        self.max_depth = 1
        self.time_per_move = 10
        self.total_moves = 0
        self.previous_board = None

    def get_move(self, board, player=1, **kwargs):
        self.player = player
        # Detect new game if board has fewer tokens than previously
        if self.previous_board is None or np.count_nonzero(board) < np.count_nonzero(self.previous_board):
            self.reset()
            logging.info("New game detected. Agent state reset.")
        self.previous_board = board.copy()
        self.total_moves += 1

        try:
            elapsed_time = time.time() - self.start_time
            remaining_time = self.TIME_LIMIT - elapsed_time
            moves_left = 42 - self.total_moves
            self.time_per_move = max(remaining_time / moves_left, 1) if moves_left > 0 else 1
            # Enforce a maximum per-move time to avoid backend timeout (e.g. 4 seconds)
            effective_time = min(self.time_per_move, 4)

            best_move = None
            self.max_depth = 1
            end_time = time.time() + effective_time

            while True:
                try:
                    move = self.iterative_deepening(board, end_time)
                    if move is not None:
                        best_move = move
                    self.max_depth += 1
                except TimeoutError:
                    logging.info("Timeout reached during iterative deepening.")
                    break
                except Exception as e:
                    logging.error(f"Error during iterative deepening: {e}")
                    break

            if best_move is None:
                valid_moves = HelperFunctions.get_valid_moves(board)
                best_move = valid_moves[0] if valid_moves else None

            return best_move

        except Exception as e:
            logging.error(f"Unexpected error in get_move: {e}")
            valid_moves = HelperFunctions.get_valid_moves(board)
            return valid_moves[0] if valid_moves else None

    def iterative_deepening(self, board, end_time):
        self.end_time = end_time
        score, move = self.alphabeta(board, depth=self.max_depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
        return move

    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        if time.time() > self.end_time:
            raise TimeoutError

        board_key = self.board_to_tuple(board)
        if board_key in self.transposition_table and self.transposition_table[board_key]['depth'] >= depth:
            entry = self.transposition_table[board_key]
            return entry['score'], entry['move']

        terminal_state = HelperFunctions.check_win(board)
        if depth == 0 or terminal_state is not None:
            eval_score = self.evaluate(board)
            return eval_score, None

        valid_moves = HelperFunctions.get_valid_moves(board)
        if not valid_moves:
            return self.evaluate(board), None

        ordered_moves = self.order_moves(valid_moves, board, maximizing_player)
        best_move = None

        if maximizing_player:
            value = float('-inf')
            for move in ordered_moves:
                child_board = HelperFunctions.place(move, board, self.player)
                score, _ = self.alphabeta(child_board, depth - 1, alpha, beta, False)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self.transposition_table[board_key] = {'score': value, 'move': best_move, 'depth': depth}
            return value, best_move
        else:
            value = float('inf')
            for move in ordered_moves:
                child_board = HelperFunctions.place(move, board, -self.player)
                score, _ = self.alphabeta(child_board, depth - 1, alpha, beta, True)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self.transposition_table[board_key] = {'score': value, 'move': best_move, 'depth': depth}
            return value, best_move

    def order_moves(self, moves, board, maximizing_player):
        center = board.shape[1] // 2
        scores = {}
        for move in moves:
            temp_board = HelperFunctions.place(move, board, self.player if maximizing_player else -self.player)
            scores[move] = self.evaluate(temp_board)
        ordered_moves = sorted(moves, key=lambda x: (scores[x], -abs(x - center)), reverse=maximizing_player)
        return ordered_moves

    def evaluate(self, board):
        terminal_state = HelperFunctions.check_win(board)
        if terminal_state == self.player:
            return float('inf')
        elif terminal_state == -self.player:
            return float('-inf')
        elif terminal_state == 0:
            return 0

        score = 0
        center_array = board[:, board.shape[1] // 2]
        center_count = int((center_array == self.player).sum())
        score += center_count * 6

        # Horizontal evaluation
        for r in range(board.shape[0]):
            row_array = board[r, :]
            for c in range(board.shape[1] - 3):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window)
        # Vertical evaluation
        for c in range(board.shape[1]):
            col_array = board[:, c]
            for r in range(board.shape[0] - 3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window)
        # Positive diagonal evaluation
        for r in range(board.shape[0] - 3):
            for c in range(board.shape[1] - 3):
                window = [board[r + i, c + i] for i in range(4)]
                score += self.evaluate_window(window)
        # Negative diagonal evaluation
        for r in range(board.shape[0] - 3):
            for c in range(board.shape[1] - 3):
                window = [board[r + 3 - i, c + i] for i in range(4)]
                score += self.evaluate_window(window)
        return score

    def evaluate_window(self, window):
        window = list(window)
        score = 0
        opp_player = -self.player
        if window.count(self.player) == 4:
            score += 100
        elif window.count(self.player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(self.player) == 2 and window.count(0) == 2:
            score += 2
        if window.count(opp_player) == 3 and window.count(0) == 1:
            score -= 4
        return score

    def board_to_tuple(self, board):
        return tuple(map(tuple, board))

###############################
# Main: Parse arguments and output move
###############################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OmegaZero AI Agent for Connect 4")
    parser.add_argument('--board', required=True, type=str,
                        help='JSON-encoded board (a 2D list)')
    parser.add_argument('--player', required=True, type=int,
                        help='Player number (1 or -1)')
    args = parser.parse_args()

    board_list = json.loads(args.board)
    board = np.array(board_list)
    player = args.player

    agent = OmegaZero()
    move = agent.get_move(board, player)
    print(move if move is not None else -1)

