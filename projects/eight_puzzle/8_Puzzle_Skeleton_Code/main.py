from board import Board
import numpy as np
import time
from agent import a_star_search, BF, MT, CB, NA

def main():

    heuristics = [BF, MT, CB, NA]
    for m in [10, 20, 30, 40]:
        for seed in range(0, 10):
            board = Board(m, seed)
            print(f"Seed: {seed} | Shuffle: {m}")
            for heuristic in heuristics:
                start = time.process_time()
                solution = a_star_search(board, heuristic)
                end = time.process_time()
                solution_cpu_time = end - start
                moves = [move[3] for move in solution[1:]]
                print(f"Heuristic: {heuristic.__name__} | Solution Length: {len(moves)} | Nodes Searched: {len(solution)} | Time: {solution_cpu_time:.6f} seconds")
                board.reset_board()

if __name__ == "__main__":
    main()

