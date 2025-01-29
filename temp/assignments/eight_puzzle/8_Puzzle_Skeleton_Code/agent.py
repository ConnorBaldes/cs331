from __future__ import annotations
from board import Board
from collections.abc import Callable
import heapq
import numpy as np

'''
Heuristics
'''
def BF(board: Board) -> int:
    return 0

def MT(board: Board) -> int:
    return np.sum(board.state != board.solution) - 1

def CB(board: Board) -> int:
    distance = 0
    for i in range(1, 9):
        pos1 = np.where(board.state == i)
        pos2 = np.where(board.solution == i)
        distance += abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    return distance[0]

def NA(board: Board) -> int:
    return MT(board) + CB(board) // 2


'''
A* Search 
'''
def a_star_search(board: Board, heuristic: Callable[[Board], int]):
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    open_set = [(heuristic(board), 0, board, None)]
    closed_set = set()
    came_from = {}

    while open_set:
        current = heapq.heappop(open_set)
        if current[2].goal_test():
            return reconstruct_path(came_from, current)

        closed_set.add(current[2])

        for neighbor, move in current[2].next_action_states():
            if neighbor in closed_set:
                continue

            tentative_g_score = current[1] + 1
            for node in open_set:
                if node[2] == neighbor and tentative_g_score >= node[1]:
                    break
            else:
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor), tentative_g_score, neighbor, current[2]))
                came_from[neighbor] = current[2]

    return None
