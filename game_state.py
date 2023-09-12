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
@dataclass
class GameState:
    board: np.ndarray = field(init=False)
    placements: List[PiecePlacement] = field(default_factory=lambda: [PiecePlacement(piece) for piece in pieces.pieces])
    _color_to_placement_index: Dict[pieces.Color, int] = field(init=False)
    def __post_init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype='uint8')
        for row_idx in range(BOARD_SIZE):
            for col_idx in range(row_idx + 1, BOARD_SIZE):
                self.board[row_idx, col_idx] = 1
        self._color_to_placement_index = {placement.piece.color: idx for idx, placement in enumerate(self.placements)}
        
    def find_placements(self) -> [PiecePlacement]:
        '''only choose ones that are touching an edge'''
        possible_placements = []
        unplaced = [copy.deepcopy(p) for p in self.placements if not p.is_placed]
        for placement in unplaced:
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
                                placement2 = PiecePlacement(
                                    piece=placement1.piece,
                                    flipped=flip,
                                    rotation=rot,
                                    row=row,
                                    col=col,
                                    is_placed=True
                                )
                                # placement2.row = row
                                # placement2.col = col
                                # placement2.is_placed = True
                                possible_placements.append(placement2)
                                # gs = self.apply_placement(placement2)
                                # if np.max(gs.board) > 1:
                                #     import pdb; pdb.set_trace()

        return possible_placements
    
    def apply_placement(self, placement: PiecePlacement) -> 'GameState':
        placement_index = self._color_to_placement_index[placement.piece.color]
        newGameState = copy.deepcopy(self)
        # newGameState.board = self.board
        if not placement.is_placed:
            import pdb; pdb.set_trace()
            return newGameState
        newGameState.placements[placement_index] = copy.copy(placement)
        shape = placement.oriented_shape()
        num_shape_rows, num_shape_cols = shape.shape
        for row in range(num_shape_rows):
            for col in range(num_shape_cols):
                newGameState.board[row + placement.row, col + placement.col] += shape[row, col]
        
        return newGameState


def apply_all_placements(placements) -> GameState:
    gs = GameState()
    gs.placements = copy.copy(placements)
    for placement in placements:
        if not placement.is_placed:
            continue
        gs = gs.apply_placement(placement=placement)
    return gs
    
@dataclass
class GameTree:
    state_action_idx_stack: List[Tuple[GameState, List[PiecePlacement], int]] = field(init=False)
    successful_placements: List[List[PiecePlacement]] = field(default_factory=list)
    
    def __post_init__(self):
        gs = GameState()
        self.state_action_idx_stack = [(gs, gs.find_placements(), 0)]

    def find_all_placements(self, game_state = GameState(), root=True):
        if all([p.is_placed for p in game_state.placements]):
            self.successful_placements.append(game_state.placements)
            print(f"found {len(self.successful_placements)} solutions at {time.time() - start_time}")
            # print([p.is_placed for p in game_state.placements])
            # print(game_state.board)
            return
        
        all_placements = game_state.find_placements()
        for idx, placement in enumerate(all_placements):
            if root:
                print(f"{idx} / {len(all_placements)} {time.time() - start_time}")
            new_game_state = game_state.apply_placement(placement)
            # print('new', len([i for i in new_game_state.placements if i.is_placed]))
            self.find_all_placements(new_game_state, root=False)
    
    # def next(self):
    #     game_state, placements, idx = self.state_action_idx_stack[-1]
    #     placements = game_state.find_placements()
    #     if len(self.state_action_idx_stack) == len(pieces.pieces):
    #         self.successful_placements.append(game_state.placements)
    #         print(f"Found {len(self.successful_placements)}th successful placement!")
    #         # return False # end early


    #     if idx >= len(placements):
    #         print(f"depth: {len(self.state_action_idx_stack)} No actions left. Popping...")
    #         print(game_state.board)
    #         self.state_action_idx_stack.pop()

    #         if len(self.state_action_idx_stack) == 0:
    #             print("Complete!")
    #             return False
    #         else: 
    #             return True

    #     if not all([a == b for a, b in zip(game_state.find_placements(), placements)]):
    #         test = [a == b for a, b in zip(game_state.find_placements(), placements)]
    #         print(game_state.board)
    #         print('first')
    #         import pdb; pdb.set_trace()

    #     test_game_state = apply_all_placements(game_state.placements)
    #     if not np.all(test_game_state.board == game_state.board):
    #         import pdb; pdb.set_trace()
        
    #     self.state_action_idx_stack.pop()
    #     self.state_action_idx_stack.append((game_state, placements, idx + 1))
    #     newGameState = game_state.apply_placement(placements[idx])
    #     self.state_action_idx_stack.append((newGameState, newGameState.find_placements(), 0))

    #     return True
        
    
import time
start_time = time.time()
gt = GameTree()
# iter_count = 1000000
# while(gt.next()):
#     print(iter_count)
#     iter_count -= 1
#     if iter_count == 0:
#         iter_count = int(input("Iterations: "))
initial_game_state = GameState()
gt.find_all_placements(initial_game_state)



# g = apply_all_placements(gt.successful_placements[0])
# print(g.board)

        
        