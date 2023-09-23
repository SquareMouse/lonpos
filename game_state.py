import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import pieces
import scipy
# import copy
from pprint import pprint
from termcolor import cprint
import sys
import time
import multiprocessing

@dataclass(slots=True)
class PiecePlacement:
    piece: pieces.Piece
    rotation: int = 0
    flipped: bool = False
    row: int = 0 # of the top left corner
    col: int = 0 # of the top left corner
    is_placed: bool = False
    
    def oriented_shape(self):
        if self.flipped:
            return np.rot90(np.fliplr(self.piece.shape), self.rotation)
        return np.rot90(self.piece.shape, self.rotation)
    def __repr__(self): 
        rotation = list('^>v<')
        return f"{self.row} {self.col} {'-' if self.flipped else '|'} {rotation[self.rotation]} {self.piece.color} "

    def __eq__(self, other):
        if self.piece.color != other.piece.color:
            return False
        if self.rotation != other.rotation or self.flipped != other.flipped or self.row != other.row or self.col != other.col:
            return False
        if self.is_placed != other.is_placed:
            return False
        return True
    
    def copy(self) -> 'PiecePlacement':
        # return copy.copy(self)
        return PiecePlacement(
            piece = self.piece,
            rotation = self.rotation,
            flipped = self.flipped,
            row = self.row,
            col = self.col,
            is_placed = self.is_placed
        )
    

BOARD_SIZE = 10
def make_board():
    board = np.ones((BOARD_SIZE+2, BOARD_SIZE+2), dtype='uint8')
    for col_idx in range(1, BOARD_SIZE+1):
        for row_idx in range(col_idx, BOARD_SIZE + 1):
            board[row_idx, col_idx] = 0
    return board
board = make_board()
_color_to_placement_index = {placement.piece.color: idx for idx, placement in enumerate([PiecePlacement(piece) for piece in pieces.pieces])}

@dataclass
class GameState:
    board: np.ndarray = field(default_factory=lambda: board.copy())
    placements: List[PiecePlacement] = field(default_factory=lambda: [PiecePlacement(piece) for piece in pieces.pieces])
    # def __post_init__(self):

        
    def find_placements(self, depth=None) -> [PiecePlacement]:
        '''only choose ones that are touching an edge'''
        possible_placements = []
        if depth is not None:
            unplaced = [self.placements[depth].copy()]
        else:
            unplaced = [p for p in self.placements if not p.is_placed]
        for placement in unplaced:
            placed = False
            for flip in placement.piece.available_flip:
                for rot in placement.piece.available_rotations:
                    placement1 = placement.copy()
                    placement1.flipped = flip
                    placement1.rotation = rot
                    shape = np.rot90(placement1.oriented_shape(), 2)  # for the convolution
                    convolved = scipy.signal.convolve2d(self.board, shape, mode='valid')
                    (num_rows, num_cols) = convolved.shape
                    for row in range(num_rows):
                        for col in range(num_cols):
                            if convolved[row, col] == 0:
                                placement2 = PiecePlacement(
                                    piece=placement1.piece,
                                    flipped=flip,
                                    rotation=rot,
                                    row=row,
                                    col=col,
                                    is_placed=True
                                )
                                possible_placements.append(placement2)
                                placed = True

            if not placed:
                return []
        return possible_placements
    
    def apply_placement(self, placement: PiecePlacement) -> 'GameState':
        placement_index = _color_to_placement_index[placement.piece.color]
        newGameState = self.copy()
        # newGameState.board = self.board
        assert placement.is_placed
        newGameState.placements[placement_index] = placement.copy()
        shape = placement.oriented_shape()
        num_shape_rows, num_shape_cols = shape.shape
        for row in range(num_shape_rows):
            for col in range(num_shape_cols):
                newGameState.board[row + placement.row, col + placement.col] |= shape[row, col]
        
        return newGameState
    
    def show(self):
        aggregate = np.zeros_like(self.board)
        for placement in self.placements:
            if not placement.is_placed:
                continue
            shape = placement.oriented_shape()
            num_shape_rows, num_shape_cols = shape.shape
            for row in range(num_shape_rows):
                for col in range(num_shape_cols):
                    if shape[row, col]:
                        aggregate[row + placement.row, col + placement.col] = placement.piece.color.value
        for row in aggregate:
            for cell in row:
                cprint("  ", "black", pieces.Color(cell).termcolor(), end="")
            print()

    def copy(self) -> 'GameState':
        return GameState(board=self.board.copy(),
                                 placements=[p.copy() for p in self.placements])

def apply_all_placements(placements, current_board: np.ndarray) -> np.ndarray:
    board = current_board.copy()

    for placement in placements:
        if not placement.is_placed:
            continue
        shape = placement.oriented_shape()
        num_shape_rows, num_shape_cols = shape.shape
        for row in range(num_shape_rows):
            for col in range(num_shape_cols):
                board[row + placement.row, col + placement.col] |= shape[row, col]
    return board




@dataclass
class GameTree:
    successful_placements: List[List[PiecePlacement]] = field(default_factory=list)

    def find_all_placements(self, game_state = GameState(), depth=0, start_time=None):
        if all([p.is_placed for p in game_state.placements]):
            self.successful_placements.append(game_state.placements)
            if start_time is not None:
                print(f"found {len(self.successful_placements)} solutions at {time.time() - start_time}")
            else:
                print("None start time")
            game_state.show()
            # if len(self.successful_placements) == 2:
            #     sys.exit()
            return
        

        all_applied_board = apply_all_placements(game_state.find_placements(), current_board=game_state.board)
        if np.min(all_applied_board) == 0:
            # print(f"Impossible at depth {depth} with len placements {len(all_placements)}")
            # print(all_applied_board)
            return

        next_piece_placements = game_state.find_placements(depth=depth)
        for idx, placement in enumerate(next_piece_placements):
            # if depth < 6:
            #     print(f"{' | ' * depth} {idx} / {len(next_piece_placements)} {time.time() - start_time}")
                
            new_game_state = game_state.apply_placement(placement)
            self.find_all_placements(new_game_state, depth=depth + 1, start_time=start_time)

def solve_game_tree(placement: PiecePlacement, depth: int, initial_game_state: GameState, start_time) -> GameTree:

    gt = GameTree()
    gt.find_all_placements(initial_game_state.apply_placement(placement), depth=depth, start_time = start_time)
    return gt

if __name__ == '__main__':
    
    start_time = time.time()


    initial_game_state = GameState()
    initial_game_state.placements[0].row = 8
    initial_game_state.placements[0].col = 6
    initial_game_state.placements[0].is_placed = True

    initial_game_state.placements[1].row = 9
    initial_game_state.placements[1].col = 1
    initial_game_state.placements[1].flipped = True
    initial_game_state.placements[1].rotation = 2
    initial_game_state.placements[1].is_placed = True

    initial_game_state.board = apply_all_placements(initial_game_state.placements, initial_game_state.board)

    starting_placements = initial_game_state.find_placements(depth=2)

    start_time = time.time()

    with multiprocessing.Pool() as p:
        args = [(sp, 3, initial_game_state, start_time) for sp in starting_placements]
        solved_game_trees = p.starmap(solve_game_tree, args)
        
    # gt.find_all_placements(initial_game_state, depth=2)



    print(f"terminated after {time.time() - start_time} seconds")
    for solved in solved_game_trees:
        for placement in solved.successful_placements:
            print(placement)


        
        