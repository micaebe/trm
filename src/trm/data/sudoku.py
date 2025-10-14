import torch
import numpy as np
from torch.utils.data import Dataset


def _find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return (i, j)
    return None

def _is_valid(board, num, pos):
    if num in board[pos[0], :]:
        return False
    if num in board[:, pos[1]]:
        return False
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    if num in board[box_y*3:box_y*3 + 3, box_x*3:box_x*3 + 3]:
        return False
    return True

def _solve_sudoku_board(board):
    find = _find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    nums_to_try = np.arange(1, 10)
    np.random.shuffle(nums_to_try)

    for num in nums_to_try:
        if _is_valid(board, num, (row, col)):
            board[row, col] = num
            if _solve_sudoku_board(board):
                return True
            board[row, col] = 0
    return False


class SudokuDataset(Dataset):
    def __init__(self, num_samples=10000, difficulty=0.5):
        super().__init__()
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.seq_len = 81

        self.puzzles = np.zeros((self.num_samples, self.seq_len), dtype=np.int64)
        self.solutions = np.zeros((self.num_samples, self.seq_len), dtype=np.int64)
        
        print(f"Generating {self.num_samples} unique Sudoku puzzles")
        self._generate_data()

    def _generate_data(self):
        for i in range(self.num_samples):
            board = np.zeros((9, 9), dtype=np.int64)
            _solve_sudoku_board(board)
            solution = board.copy()
            
            puzzle = solution.flatten()
            mask = np.random.rand(self.seq_len) < self.difficulty
            puzzle[mask] = 0
            
            self.puzzles[i] = puzzle
            self.solutions[i] = solution.flatten()
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        puzzle = torch.from_numpy(self.puzzles[idx])
        solution = torch.from_numpy(self.solutions[idx])
        return puzzle, solution

