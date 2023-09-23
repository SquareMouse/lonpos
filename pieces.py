
from enum import Enum, auto
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import List
import termcolor

class Color(Enum):
    black = 0
    red  = 1
    magenta  = 2
    mistblue  = 3
    skyblue  = 4
    blue  = 5
    purple  = 6
    pink  = 7
    limegreen  = 8
    green  = 9
    orange  = 10
    yellow  = 11
    white  = 12

    def termcolor(self):
        return {
            Color.black: "on_black",
            Color.red: "on_red",
            Color.magenta: "on_magenta",
            Color.mistblue: "on_light_cyan",
            Color.skyblue: "on_light_blue",
            Color.blue: "on_blue",
            Color.purple: "on_light_magenta",
            Color.pink: "on_light_red",
            Color.limegreen: "on_light_green",
            Color.green: "on_green",
            Color.orange: "on_light_yellow", # wack
            Color.yellow: "on_yellow",
            Color.white: "on_white",
        }[self]
     

@dataclass(slots=True)
class Piece:
    shape: np.ndarray
    color: Color
    available_rotations: list[int] = field(default_factory=lambda : list(range(4)))
    available_flip: List[bool] = field(default_factory=lambda : [False, True])
    
    @cached_property
    def size(self):
        return np.sum(self.shape)
    

red = Piece(color=Color.red,
    shape=np.array([
        1,0,
        1,1,
        1,1,
    ]).astype('uint8').reshape((3,2))
)

# red = Piece(color=Color.red,
#             shape=np.array([[1]]),
#             available_flip=[False], available_rotations=[0])

magenta = Piece(color=Color.magenta,
    shape=np.array([
        1,0,0,
        1,1,0,
        0,1,1,
    ]).astype('uint8').reshape((3,3)),
    available_flip=[False]
)

purple = Piece(color=Color.purple,
            shape=np.array([[1,1,1,1]]),
            available_rotations=list(range(2)),
            available_flip=[False])

pink = Piece(color=Color.pink, shape=np.array([
    1,1,1,1,
    0,0,1,0,
]).astype('uint8').reshape((2,4)))
yellow = Piece(color=Color.yellow, 
    shape=np.array([
        1,1,1,
        1,0,1,
    ]).astype('uint8').reshape((2,3)),
    available_flip=[False])
sky_blue = Piece(color=Color.skyblue, 
    shape=np.array([
        1,1,1,
        0,0,1,
        0,0,1,
    ]).astype('uint8').reshape((3,3)),
    available_flip=[False])
mist_blue = Piece(color=Color.mistblue,
    shape=np.array([
        0,1,0,
        1,1,1,
        0,1,0,
    ]).astype('uint8').reshape((3,3)),
    available_rotations=[0],
    available_flip=[False])

white = Piece(color=Color.white,
    shape=np.array([
        1,0,
        1,1,
    ]).astype('uint8').reshape((2,2)),
    available_flip=[False])
orange = Piece(color=Color.orange,
    shape=np.array([
        1,1,1,
        1,0,0,
    ]).astype('uint8').reshape((2,3)))

green = Piece(color=Color.green,
    shape=np.array([
        1,1,1,0,
        0,0,1,1,
    ]).astype('uint8').reshape((2,4)))

blue = Piece(color=Color.blue,
    shape=np.array([
        1,0,0,0,
        1,1,1,1,
    ]).astype('uint8').reshape((2,4)))

lime_green = Piece(color=Color.limegreen,
    shape=np.array([
        1,1,
        1,1
    ]).astype('uint8').reshape((2,2)),
    available_flip=[False],
    available_rotations=[0])

pieces = [
    green,
    pink,
    mist_blue,
    magenta,
    blue,
    purple,
    sky_blue,
    red,
    yellow,
    orange,
    lime_green,
    white,
]