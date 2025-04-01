from typing import Union

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

import asyncio

import numpy as np

import json

import os

from GUI.RL_puzzle_solver.HIL.puzzle_solver import puzzle_solver, assemble
from GUI.Back_End import BackEnd

# app = FastAPI()
    
pieces_names_g29: list = ['RPf_00204', 'RPf_00205', 'RPf_00206', 'RPf_00207', 'RPf_00208']

class G29(BaseModel):
    name: str = 'group_29'
    P_x: int = 101
    P_y: int = 101
    P_theta: int = 90 
    pieces_names: list = ['RPf_00204', 'RPf_00205', 'RPf_00206', 'RPf_00207', 'RPf_00208']
    pieces: dict = {'pieces':{}}

class PuzzlePieces(BaseModel):
    pieces: dict = {'pieces': {}}

class API(FastAPI):
    def __init__(self, update_interval):
        super().__init__()
        self.path_dic = {}
        self.update_interval = update_interval
        self.back_end = BackEnd()
        self.set_paths()
        self.get("/")(self.read_root)
        self.get("/solve")(self.solve_puzzle)

        self.get("/results")(self.get_results)

        # self.websocket("/ws")(self.websocket_endpoint)
        # self.websocket("/ws")(self.websocket_endpoint)

    def set_paths(self):
        path_dic = self.back_end.setting()


    def read_root(self, number):
        return {"Hello": "World"}

    def get_results(self):
        print(self.back_end.pl_solver_running)
        if self.back_end.pl_solver_running:
            answer, probability, process = self.back_end.get_API_solution()
            data = self.save_parameters_to_json(answer, probability, process)
            if data is not None:
                return data
            else:
                return "error 404"
        return "not running"

    def solve_puzzle(self, key_fragment):
        xy_step, theta_step = self.back_end.extract_steps()

        print(xy_step, theta_step)

        image_path = self.back_end.path_dic["image_path"]

        image_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]


        print(image_names)

        if not (key_fragment in image_names):
            answer = "choose from this list"
            answer = answer + str(image_names)
            return answer
        else:
            neighbour_ids = image_names.copy()
            neighbour_ids.remove(key_fragment)
            print('neighbour', neighbour_ids)
            self.back_end.start_pl_solver_thread(key_fragment, neighbour_ids, [])

    # async def websocket_endpoint(self, websocket: WebSocket):
    #     await websocket.accept()
    #     while True:
    #         if self.back_end.pl_solver_running:
    #             answer, probability, process = self.back_end.get_API_solution()
    #             data = self.save_parameters_to_json(answer, probability, process)
    #             if data is not None:
    #                 await websocket.send_json(data)
    #             else:
    #                 await websocket.send_text("error 404")
    #         await asyncio.sleep(self.update_interval)
    #     await websocket.close()
    #
    # async def websocket_endpoint(self, websocket: WebSocket):
    #     await websocket.accept()
    #     try:
    #         while True:
    #             key_fragment = await websocket.receive_text()
    #             print(f"Received: {key_fragment}")
    #
    #             image_path = self.path_dic["image_path"]
    #             image_names = [os.path.splitext(f)[0] for f in os.listdir(image_path) if
    #                            f.endswith(('.jpg', '.png'))]
    #
    #             if key_fragment not in image_names:
    #                 await websocket.send_text("Key Fragment not found")
    #             else:
    #                 neighbour_ids = image_names.copy()
    #                 neighbour_ids.remove(key_fragment)
    #                 await websocket.send_text("Solver started")
    #
    #                 # Simulate processing with progress updates
    #                 for i in range(1, 6):
    #                     await websocket.send_text(f"Processing {key_fragment}: Step {i}/5")
    #                     await asyncio.sleep(2)  # Simulate step execution
    #
    #                 self.back_end.start_pl_solver_thread(key_fragment, neighbour_ids, [])
    #                 await websocket.send_text(f"Processing for {key_fragment} completed.")
    #     except:
    #         await websocket.close()






        # self.back_end.start_pl_solver_thread(last_loop_solution=app.final_solution)
        #
        # app.final_solution = back_end.loop_finalization(solved_pieces, app.image_offset)
        #
        # answer, probability, process = back_end.get_solution_dict()
        #
        # back_end.set_path(path_dic)
        #
        # pos = back_end.reverse_offset(pos, offset)
        # pos = back_end.scale_to_solver(xy_step, theta_step, pos, path_dic)
        # P = positions_to_P(..)
        # P_ = update_P(..)
        # positions_ = P_to_positions(..)
        # response = {'pieces': {'RPf_XXX': {'position': [x, y, z], 'probability': 0.1}, 'RPf_XXX': {'position': [x, y, z], 'probability': 0.1}, .. } }
        response = "Will Do"
        return xy_step, theta_step

    def save_parameters_to_json(self, answer, probability, process, filename="API-example.json"):
        # Convert numpy arrays to lists for JSON serialization
        if answer is not None and probability is not None and process is not None:
            data = {
                "pieces": {
                    key: {
                        "position": value.tolist(),
                        "probability": probability[key].tolist()
                    } for key, value in answer.items()
                },
                "process": process
            }
            return data  # Returning dictionary (FastAPI auto-converts to JSON)
        return None

interval = 2 # in seconds
app = API(interval)

# @app.get('/random')
# def random_positions(puzzle: str):
#     if puzzle == 'group_29':
#         group = G29()
#         # print(group)
#         for pname in group.pieces_names:
#             group.pieces["pieces"][pname] = {"position": np.random.uniform(0, 1, 3).tolist(), "probability": 0}
#     else:
#         return "Not done yet, please use `group_29` as puzzle"
#     return group.pieces
#
# @app.get('/solve')

#
#
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}
#
# @app.get("/multiply/{item_id}")
# def read_item(item_id: int, k: int, q: Union[str, None] = None):
#     res = item_id * k
#     return {"item_id": item_id, "q": q, "result": res}
#
#
# # @app.put("/items/{item_id}")
# # def update_item(item_id: int, item: Item):
# #     return {"item_name": item.name, "item_id": item_id}