import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import pieces
import scipy
import copy
from pprint import pprint

@dataclass
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

        
    def find_placements(self) -> [PiecePlacement]:
        '''only choose ones that are touching an edge'''
        possible_placements = []
        unplaced = [copy.deepcopy(p) for p in self.placements if not p.is_placed]
        for placement in unplaced:
            placed = False
            for flip in placement.piece.available_flip:
                for rot in placement.piece.available_rotations:
                    placement1 = PiecePlacement(
                        piece=placement.piece   
                    )
                    copy.deepcopy(placement)
                    placement1.flipped = flip
                    placement1.rotation = rot
                    shape = np.rot90(placement1.oriented_shape(), 2)  # for the convolution
                    convolved = scipy.signal.convolve2d(self.board, shape, mode='valid')
                    (num_rows, num_cols) = convolved.shape
                    for row in range(num_rows):
                        for col in range(num_cols):
                            if convolved[row, col] == 0:
                                # ok = False
                                # for (row_diff, col_diff) in [(r,c) for r in [-1,1] for c in [-1,1]]:
                                #     if convolved[row + row_diff, col + col_diff] != 0:
                                #         ok = True
                                #         break
                                # if ok == False:
                                #     continue
                                        
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
        newGameState = copy.deepcopy(self)
        # newGameState.board = self.board
        assert placement.is_placed
        newGameState.placements[placement_index] = copy.copy(placement)
        shape = placement.oriented_shape()
        num_shape_rows, num_shape_cols = shape.shape
        for row in range(num_shape_rows):
            for col in range(num_shape_cols):
                newGameState.board[row + placement.row, col + placement.col] |= shape[row, col]
        
        return newGameState


def apply_all_placements(placements) -> GameState:
    gs = GameState()
    gs.placements = copy.copy(placements)
    for placement in placements:
        if not placement.is_placed:
            continue
        gs = gs.apply_placement(placement=placement)
    return gs

empty_cell_finders = [
    np.array([
        [0,1,0],
        [1,-1,1],
        [0,1,0],
    ]),
    np.array([
        [0,1,1,0],
        [1,-1,-1,1],
        [0,1,1,0],
    ]),
    np.array([
        [0,1,1,0],
        [1,-1,-1,1],
        [0,1,1,0],
    ]).T,
]

@dataclass
class GameTree:
    successful_placements: List[List[PiecePlacement]] = field(default_factory=list)

    def find_all_placements(self, game_state = GameState(), depth=0):
        for limit, empty_cell_finder in zip([4,6,6], empty_cell_finders):
            if np.max(scipy.signal.convolve2d(empty_cell_finder, game_state.board)) == limit:
                return
        if all([p.is_placed for p in game_state.placements]):
            self.successful_placements.append(game_state.placements)
            print(f"found {len(self.successful_placements)} solutions at {time.time() - start_time}")
            return
        
        # if depth <= 5:
        #     print([p.is_placed for p in game_state.placements])
        #     print(game_state.board)
        
        all_placements = game_state.find_placements()

        if depth <= 5:
            if np.min(apply_all_placements(all_placements).board) == 0:
                print(f"Impossible at depth {depth}")
                return

        for idx, placement in enumerate(all_placements):
            if depth < 6:
                print(f"{' | ' * depth} {idx} / {len(all_placements)} {time.time() - start_time}")
            new_game_state = game_state.apply_placement(placement)
            self.find_all_placements(new_game_state, depth=depth + 1)

        
    
import time
start_time = time.time()
gt = GameTree()

initial_game_state = GameState()
gt.find_all_placements(initial_game_state)




# g = apply_all_placements(gt.successful_placements[0])
# print(g.board)

        
        