from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np 

app = FastAPI()
    
pieces_names_g29: list = ['RPf_00204', 'RPf_00205', 'RPf_00206', 'RPf_00207', 'RPf_00208']

class G29(BaseModel):
    name: str = 'group_29'
    P_x: int = 101
    P_y: int = 101
    P_theta: int = 90 
    pieces_names: list = ['RPf_00204', 'RPf_00205', 'RPf_00206', 'RPf_00207', 'RPf_00208']
    pieces: dict = {'pieces':{}}

    def __init__(self, pieces: dict):
        self.pieces = pieces
    
    def run_update_loop(self, num_iterations: int = 100):
        self.run_update_loop # NO

class PuzzlePieces(BaseModel):
    pieces: dict = {'pieces': {}}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/random')
def random_positions(puzzle: str):
    if puzzle == 'group_29':
        group = G29()
        for pname in group.pieces_names:
            group.pieces["pieces"][pname] = {"position": np.random.uniform(0, 1, 3).tolist(), "probability": 0}
    else:
        return "Not done yet, please use `group_29` as puzzle"
    return group.pieces

@app.get('/solve')
def solve_puzzle(puzzle: str, anchor: int, pieces: PuzzlePieces):
    # P = positions_to_P(..)
    # P_ = update_P(..)
    # positions_ = P_to_positions(..)
    # response = {'pieces': {'RPf_XXX': {'position': [x, y, z], 'probability': 0.1}, 'RPf_XXX': {'position': [x, y, z], 'probability': 0.1}, .. } }
    return 'will do'


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/multiply/{item_id}")
def read_item(item_id: int, k: int, q: Union[str, None] = None):
    res = item_id * k
    return {"item_id": item_id, "q": q, "result": res}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}