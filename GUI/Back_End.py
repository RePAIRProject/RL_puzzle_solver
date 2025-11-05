import time
import warnings
from threading import Thread, Event, Lock

import yaml

import RL_puzzle_solver.parameters_utils as yaml_config
import paramiko
import getpass
from scipy.io import loadmat
from scipy.spatial import KDTree


import sys
import os

from GUI.Evaluation import Evaluation
from GUI.RL_puzzle_solver.configs.folder_names import ground_truth

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import select_anchor_RePAIR as select_anchor_RePAIR
import RL_puzzle_solver.HIL.puzzle_solver as puzzle_solver
import json
import numpy as np
import re
from pathlib import Path
# from kivymd.app import MDApp
import math

class BackEnd:
    def __init__(self):
        self.main_app = None
        self.select_anchor_running = None
        self.select_neighbour_running = None
        self.pl_solver_running = None

        self.select_anchor_thread = Thread()
        self.select_neighbour_thread = Thread()
        self.pl_solver_thread = Thread()

        self.select_anchor_lock = Lock()
        self.select_neighbour_lock = Lock()
        self.pl_solver_lock = Lock()

        self.anchor_images = []
        self.neighbour_images = []
        self.key_fragments = []

        self.image_names = []
        self.image_scores = []
        self.image_numbers = 0

        self.select_anchor_done = False
        self.select_neighbour_done = False
        self.pl_solver_done = False

        self.cache_path = None

        self.key_fragment = ""
        self.solved_pieces = []
        self.neighbour_ids = []
        self.input_dict = {}
        self.pl_solution = {}
        self.path_dic = {}

        self.backend_path = ""
        self.path = ""
        self.path_bw = ""

        self.path_dic = None
        self.sorted_neighbour_images = None
        self.neighbour_after = 0
        self.interaction_counter = 0
        self.yaml = None

        self.mat = None
        self.R = None

        self.evaluation = Evaluation()

    def set_path(self, path_dic):
        self.path_dic = path_dic

        self.backend_path = self.path_dic['backend_path']
        self.path = self.path_dic['image_path']
        self.path_bw = self.path_dic['mask_path']

        comp_folder = path_dic['comp_folder']
        comp_name = path_dic['comp_name']

        self.mat = np.load(os.path.join(comp_folder, comp_name), allow_pickle=True).item()
        self.R = self.mat[path_dic['comp_format']]

    def set_main_app(self, app):
        self.main_app = app

    def select_anchor_thread_function(self):
        self.set_select_anchor_running(True)
        anchor_images = select_anchor_RePAIR.select_anchor(self.path_dic)
        self.extract_lists(anchor_images)
        self.set_select_anchor_running(False)
        self.set_select_anchor_done(True)
        self.main_app.show_anchors()

    def removed_from_locked(self, main_image):
        main_image_name = main_image.name
        if main_image_name == self.key_fragment:
            print("YOU CANNOT REMOVE MAIN ANCHOR")
            return
        if main_image.is_anchor:
            main_image.set_anchor(False)
        self.interaction_counter += 1
        self.logger(
            "Human interacted " + str(self.interaction_counter) + " times, denied a position of the piece " + str(main_image_name))
        puzzle_solver.removed_from_locked(main_image_name)

    def set_cm_elements(self, main, neighbour, current_image_pos, image_pos, offset, value):
        xy_step, theta_step = self.extract_steps()
        pos = self.reverse_offset(current_image_pos, offset)
        pos = self.scale_to_solver(xy_step, theta_step, pos, self.path_dic)
        scaled_current_pos = pos

        pos = self.reverse_offset(image_pos, offset)
        pos = self.scale_to_solver(xy_step, theta_step, pos, self.path_dic)
        scaled_neighbour_pos = pos

        # calculate relative position
        x = scaled_current_pos[0] - scaled_neighbour_pos[0]
        y = scaled_current_pos[1] - scaled_neighbour_pos[1]
        z = scaled_current_pos[2] - scaled_neighbour_pos[2]
        relative_position = (x, y, z)

        print("scaled_current_pos", scaled_current_pos)
        print("scaled_neighbour_pos", scaled_neighbour_pos)
        print("relative_position", relative_position)

        # if value:
        #     self.logger("Human interacted " + str(self.interaction_counter) + " times, accepted a relative position of the piece " + str(main))
        # else:
        #     self.logger("Human interacted " + str(self.interaction_counter) + " times, denied a position of the piece " + str(main))

        puzzle_solver.set_cm_element(main, neighbour, relative_position, value)

    def set_p_elements(self, couple, offset, value = True):
        image_name = couple[0]
        image_pos = couple[1]
        pos = image_pos

        xy_step, theta_step = self.extract_steps()

        # pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
        pos = self.reverse_offset(pos, offset)

        pos = self.scale_to_solver(xy_step, theta_step, pos, self.path_dic)

        if value:
            puzzle_solver.set_p_elements(pos[0], pos[1], pos[2], image_name)
            self.interaction_counter += 1
            self.logger(
                "Human Interacted " + str(self.interaction_counter) + " times, moved piece (" + str(
                    image_name) + ") to:   " + "x: " + str(pos[0]) + " y: " + str(
                    pos[1]) + " tetha: " + str(pos[2]))
        else:
            puzzle_solver.set_p_elements(pos[0], pos[1], pos[2], image_name, value)
            self.logger(
                "Human Interacted, unlocked piece (" + str(image_name) + ") from fixed position:   " + "x: " + str(
                    pos[0]) + " y: " + str(
                    pos[1]) + " tetha: " + str(pos[2]))


    def reverse_offset(self, pos, offset):
        pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
        return pos

    def evaluate(self, answer):
        pieces, results = self.extract_pieces(answer)
        ground_truth_path = self.path_dic['ground_truth']
        ground_truth = self.extract_ground_truth(ground_truth_path)
        path_lists = self.extract_path_lists(pieces, self.path_dic['pieces_path'])

        #make both results absolute to one first fragment in ground_truth
        ground_truth, results = self.normalize_results_gt(results, ground_truth)

        q_pos, rmse_rot, rmse_translation = -13.0, -13.0, -13.0

        # if results != {} and ground_truth != {}:
        #     q_pos, rmse_rot, rmse_translation = self.evaluation.evaluate(pieces, ground_truth, results, path_lists)

        return q_pos, rmse_rot, rmse_translation

    def normalize_results_gt(self, results, ground_truth):
        """
        Shift & optionally rotate *results* so its bounding‑box origin and the
        orientation of the first common fragment match those of *ground_truth*.

        The function returns a *new* results dict; the originals are untouched.
        """

        base_fragment = self.key_fragment
        base_fragment = re.search(r'\d+', base_fragment)
        key_id = str(base_fragment.group(0)).zfill(5)  # zero-pad to 5 digits if needed

        if key_id not in results or key_id not in ground_truth:
            raise KeyError(f"'{key_id}' missing in results or ground_truth")

            # --- 1) translate so key fragment sits at (0,0) in each frame --------
        gx, gy, g_theta = ground_truth[key_id]
        rx, ry, r_theta = results[key_id]

        gt = {pid: [x - gx, y - gy, theta] for pid, (x, y, theta) in ground_truth.items()}
        res = {pid: [x - rx, y - ry, theta] for pid, (x, y, theta) in results.items()}

        # --- 2) shift both so GT bbox min(x,y)=0,0 ---------------------------
        min_x = min(p[0] for p in gt.values())
        min_y = min(p[1] for p in gt.values())
        dx = -min_x if min_x < 0 else 0.0
        dy = -min_y if min_y < 0 else 0.0

        gt = {pid: [x + dx, y + dy, theta] for pid, (x, y, theta) in gt.items()}
        res = {pid: [x + dx, y + dy, theta] for pid, (x, y, theta) in res.items()}

        # --- 3) rotate results so key fragment’s theta matches GT theta -------------
        d_theta = (g_theta - r_theta) % 360
        if d_theta:
            sin_t, cos_t = math.sin(math.radians(d_theta)), math.cos(math.radians(d_theta))
            res_rot = {}
            for pid, (x, y, theta) in res.items():
                x2 = cos_t * x - sin_t * y
                y2 = sin_t * x + cos_t * y
                res_rot[pid] = [x2, y2, (theta + d_theta) % 360]
            res = res_rot

        return gt, res

    def save_results(self, answer: dict[str, list | tuple | float]) -> None:
        """
        Shift all (x, y) so the most‑negative fragment sits at (0, 0) and write
        ',rpf,x,y,rot' text to <cache_path>/solution.txt.
        """
        cache_path = self.path_dic['cache_path']
        out_path   = Path(os.path.join(cache_path, "solution.txt"))

        # ---------------------------------------------------------------
        # 1) compute global min_x, min_y
        # ---------------------------------------------------------------
        xs = [float(v[0]) for v in answer.values()]
        ys = [float(v[1]) for v in answer.values()]
        shift_x = -min(xs) if min(xs) < 0 else 0.0
        shift_y = -min(ys) if min(ys) < 0 else 0.0

        # ---------------------------------------------------------------
        # 2) deterministic ordering by numeric part of the key
        # ---------------------------------------------------------------
        def numeric(k: str) -> int:
            m = re.search(r"(\\d+)", k)
            return int(m.group(1)) if m else 0
        sorted_keys = sorted(answer.keys(), key=numeric)

        # ---------------------------------------------------------------
        # 3) write file
        # ---------------------------------------------------------------
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(",rpf,x,y,rot\n")                 # header with leading ,

            for idx, key in enumerate(sorted_keys, start=0):  # 0‑based index
                x, y, rot = (float(v) for v in answer[key])
                x += shift_x
                y += shift_y

                # normalise key to 'RPf_00001'
                if key.startswith("RPf_"):
                    rpf = key.split("_mesh")[0]
                else:
                    rpf = f"RPf_{int(key):05d}"

                fh.write(f"{idx},{rpf},{x},{y},{rot}\n")

        print(f"saved {len(sorted_keys)} rows → {out_path} "
              f"(shift_x={shift_x}, shift_y={shift_y})")

    def extract_pieces(self, answer):
        """
            Takes a dictionary of full piece names with position arrays,
            and returns:
            - a list of normalized piece IDs (e.g. ['00018', '00019'])
            - a dictionary mapping these IDs to their arrays

            Normalization extracts the numeric part from the piece name.
            """
        pieces = []
        results = {}

        for full_name, array in answer.items():
            match = re.search(r'\d+', full_name)
            if match:
                piece_id = match.group(0).zfill(5)  # Normalize to 5-digit string
                pieces.append(piece_id)
                results[piece_id] = array.tolist()  # Convert NumPy array to list

        return pieces, results

    def extract_ground_truth(self, ground_truth_path):
        "ground truth should be .txt format"
        ground_truth = {}

        with open(ground_truth_path, 'r') as file:
            lines = file.readlines()

            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    raw_piece_name = parts[1]
                    # Extract the first sequence of digits (e.g., '02313')
                    match = re.search(r'\d+', raw_piece_name)
                    if not match:
                        continue  # Skip if no numeric ID found
                    piece_id = match.group(0).zfill(5)  # zero-pad to 5 digits if needed
                    x = float(parts[2])
                    y = float(parts[3])
                    rot = float(parts[4])
                    ground_truth[piece_id] = [x, y, rot]

        return ground_truth

    def extract_path_lists(self, pieces, path):
        path_lists = {}
        all_files = os.listdir(path)

        for piece_id in pieces:
            # Construct part of the filename to search for (e.g., "00232")
            matching_file = next(
                (f for f in all_files if piece_id in f and f.lower().endswith(('.png', '.jpg'))),
                None
            )
            if matching_file:
                path_lists[piece_id] = os.path.join(path, matching_file)
            else:
                path_lists[piece_id] = None  # Or skip, or raise an error

        return path_lists

    def extract_parameters(self):
        cmp_parameter_path = self.path_dic['parameters']
        ppars_yaml = {}
        print("Opening YAML file:", cmp_parameter_path)
        with open(cmp_parameter_path, 'r') as file:
            ppars_yaml = yaml.safe_load(file)

        ppars = {}
        ppars["xy_step"] = ppars_yaml['grid_params']['xy_step']
        ppars["theta_step"] = ppars_yaml['grid_params']['theta_step']
        return ppars

    def extract_steps(self):
        ppars = self.extract_parameters()
        return ppars["xy_step"], ppars["theta_step"]

    def get_iteration(self):
        iteration = puzzle_solver.get_iteration()
        return iteration

    def get_solution_dict(self):
        answer, probability, process, iteration = puzzle_solver.get_solution_dict()

        if answer is not None:
            answer = self.scale_solution(answer)
        return answer, probability, process, iteration

    def get_API_solution(self):
        answer, probability, process, iteration = puzzle_solver.get_solution_dict()
        # answer = self.scale_solution(answer)
        return answer, probability, process, iteration

    def solver_toggle_lock(self, value):
        self.logger("solver_toggle_lock:   " + str(value))
        puzzle_solver.toggle_lock(value)

    def pl_solver_thread_function(self):
        self.set_pl_solver_running(True)

        initial_thresh = 0.0
        average_thresh_factor = 0.05
        min_remaining = 2
        puzzle_solver.set_running(True)
        self.pl_solution, probability = puzzle_solver.assemble(self.input_dict, self.path_dic)
        puzzle_solver.set_running(False)
        self.pl_solution = self.throw_away_1(self.pl_solution, probability, initial_thresh)
        # pl_solution = throw_away_2(pl_solution, probability, average_thresh_factor)
        # pl_solution = combined_throw_away(pl_solution, probability, initial_thresh, min_remaining, average_thresh_factor)
        self.set_pl_solver_running(False)
        self.set_pl_solver_done(True)
        if self.main_app is not None:
            self.main_app.show_solutions()

    def throw_away_1(self, solution, probability, thresh_hold):
        for key in list(solution.keys()):
            prob = probability[key][0]
            if prob < thresh_hold:
                del solution[key]
        return solution

    def throw_away_2(self, solution, probability, thresh_hold):
        filtered_probs = [prob[0] for key, prob in probability.items() if prob[0] != 1]
        average_prob = sum(filtered_probs) / len(filtered_probs)
        threshold_value = average_prob * thresh_hold
        for key in list(solution.keys()):
            prob = probability[key][0]
            if prob != 1 and (prob - average_prob) <= threshold_value:
                del solution[key]
        return solution

    def combined_throw_away(self, solution, probability, initial_thresh, base_min_remaining, average_thresh_factor):
        kept_solutions = {}
        deleted_solutions = {}

        for key in solution.keys():
            prob = probability[key][0]
            if prob >= initial_thresh:
                kept_solutions[key] = solution[key]
            else:
                deleted_solutions[key] = solution[key]

        count_prob_1 = sum(1 for key in probability if probability[key][0] == 1)
        required_min_remaining = base_min_remaining + count_prob_1 + 1

        remaining_probs = []

        for key in kept_solutions:
            prob = probability.get(key, [None])[0]
            if prob is not None and prob != 1:
                remaining_probs.append(prob)


        if len(remaining_probs) < required_min_remaining:
            all_deleted_probs = {key: probability[key][0] for key in deleted_solutions if probability[key][0] != 1}
            all_probs = list(all_deleted_probs.values())

            if all_probs:
                average_prob = sum(all_probs) / len(all_probs)

                sorted_deleted = sorted(
                    all_deleted_probs.items(),
                    key=lambda item: item[1] - average_prob,
                    reverse=True
                )

                for key, prob in sorted_deleted:
                    if len(remaining_probs) + len(kept_solutions) >= required_min_remaining:
                        break
                    kept_solutions[key] = deleted_solutions.pop(key)
                    remaining_probs.append(prob)

        return kept_solutions

    def puzzle_solver_test_function(self, last_loop_solution, neighbour_test):
        self.input_dict.update({'solved_pieces': last_loop_solution})
        self.input_dict.update({'neighbours': neighbour_test})

        self.pl_solution = puzzle_solver.assemble(self.input_dict, self.path_dic)

    def get_next_neighbour(self, image_id):
        boolean = False
        next_neighbour = None
        neighbour_number = self.path_dic['number_of_neighbours'] + self.neighbour_after
        if self.sorted_neighbour_images is not None:
            if neighbour_number >= len(self.sorted_neighbour_images):
                self.neighbour_after = -1 * self.path_dic['number_of_neighbours']
            neighbour_number = self.path_dic['number_of_neighbours'] + self.neighbour_after
            next_neighbour = self.sorted_neighbour_images[neighbour_number]
        for image in self.neighbour_images:
            if image_id == image[0]:
                self.neighbour_images.remove(image)
                self.neighbour_images.append(next_neighbour)
                self.neighbour_after += 1
                self.extract_lists(self.neighbour_images)
                boolean = True
        return boolean, self.neighbour_images

    def select_neighbour_thread_function(self):
        self.set_select_neighbour_running(True)

        self.sorted_neighbour_images = select_anchor_RePAIR.select_neighbour(self.path_dic, self.key_fragments)

        neighbour_numbers = self.path_dic['number_of_neighbours']
        self.neighbour_images = self.sorted_neighbour_images[:neighbour_numbers]

        self.extract_lists(self.neighbour_images)

        self.set_select_neighbour_running(False)
        self.set_select_neighbour_done(True)

        self.main_app.show_neighbours()

    def extract_lists(self, main_list):
        self.image_names = []
        self.image_scores = []
        self.image_numbers = 0
        for i in range(len(main_list)):
            self.image_names.append(main_list[i][0])
            self.image_scores.append(main_list[i][1])
            self.image_numbers += 1

    def loop_finalization(self, solved_list, offset):
        final_solution = []

        xy_step, theta_step = self.extract_steps()

        for pieces in solved_list:
            name = pieces[0]
            pos = pieces[1]

            # pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
            pos = self.reverse_offset(pos, offset)

            pos = self.scale_to_solver(xy_step, theta_step, pos, self.path_dic)

            pieces[1][0] = pos[0]
            pieces[1][1] = pos[1]
            pieces[1][2] = pos[2]
            final_solution.append(pieces)
        return final_solution

    def scale_to_solver(self, xy_step, theta_step, position, dic):
        y, x, rotation = position
        x = -1 * x / xy_step
        y = y / xy_step
        x = round(x)  # should I
        y = round(y)  # should I?
        rotation = rotation / theta_step

        comp_folder = dic['comp_folder']
        comp_name = dic['comp_name']

        # comp_name = eval("f'{}'".format(comp_name))
        mat = np.load(os.path.join(comp_folder, comp_name), allow_pickle=True).item()

        R = mat[dic['comp_format']]

        bias = R.shape[0] + 1

        position = [x + bias, y + bias, rotation]
        return position

    def kill_puzzle_solver(self):
        if self.pl_solver_running:
            puzzle_solver.solver_alive(False)
            print("SOLVER HAS BEEN STOPPED")

    def start_anchor_thread(self):
        self.select_anchor_thread = Thread(target=self.select_anchor_thread_function, daemon=True)
        self.select_anchor_thread.start()

    def start_neighbour_thread(self, fragments):
        self.key_fragments = fragments
        self.select_neighbour_thread = Thread(target=self.select_neighbour_thread_function, daemon=True)
        self.select_neighbour_thread.start()

    def start_pl_solver_thread(self, key_fragment, neighbour_ids, solved_pieces):
        print('key_fragment', key_fragment)
        print('neighbour_ids', neighbour_ids)
        self.input_dict = {'anchor': key_fragment, 'neighbours': neighbour_ids, 'solved_pieces': solved_pieces}

        print("input_dict", self.input_dict)
        select_pl_solver = Thread(target=self.pl_solver_thread_function, daemon=True)
        self.interaction_counter = 0
        self.logger("dataset_name: " + self.path_dic['dataset_name'] + "   comp_name: " + self.path_dic[
            'comp_name'] + "   key_fragment: " + str(key_fragment))
        self.logger(str(len(neighbour_ids)) + " pieces are getting solved, " + str(len(solved_pieces) + 1) + " solved in previous runs")
        self.logger("id of the neighbours: " + ''.join(
            str(x) for x in neighbour_ids) + "\n" + "id of the anchors/solved pieces: " + ''.join(
            str(x) for x in solved_pieces) + " anchor_piece:" + str(key_fragment))
        select_pl_solver.start()

    def get_select_anchor_running(self):
        with self.select_anchor_lock:
            answer = self.select_anchor_running
        return answer

    def get_select_neighbour_done(self):
        with self.select_anchor_lock:
            answer = self.select_neighbour_done
        return answer

    def get_select_neighbour_running(self):
        with self.select_neighbour_lock:
            answer = self.select_anchor_running
        return answer

    def get_pl_solver_running(self):
        with self.pl_solver_lock:
            answer = self.pl_solver_running
        return answer

    def get_pl_solver_done(self):
        with self.pl_solver_lock:
            answer = self.pl_solver_done
        return answer

    def set_select_neighbour_done(self, boolean):
        with self.select_neighbour_lock:
            self.select_neighbour_done = boolean

    def set_select_anchor_running(self, boolean):
        with self.select_anchor_lock:
            self.select_anchor_running = boolean

    def set_select_anchor_done(self, boolean):
        with self.select_anchor_lock:
            self.select_anchor_done = boolean

    def set_pl_solver_done(self, boolean):
        with self.pl_solver_lock:
            self.pl_solver_done = boolean

    def set_select_neighbour_running(self, boolean):
        with self.select_neighbour_lock:
            self.select_neighbour_running = boolean

    def set_pl_solver_running(self, boolean):
        with self.pl_solver_lock:
            self.pl_solver_running = boolean

    def get_select_anchor_done(self):
        with self.select_anchor_lock:
            answer = self.select_anchor_done
        return answer

    def get_pl_solution(self):
        apply_gt = self.path_dic['apply_gt']
        if apply_gt == "True":
            self.pl_solution = self.apply_ground_truth()
        original_pl_solution = self.pl_solution.copy()
        self.pl_solution = self.scale_solution(self.pl_solution)
        return self.pl_solution, original_pl_solution

    def scale_solution(self, solution):
        key_x, key_y, key_rotation = solution[self.key_fragment]

        adjusted_solution = {}

        for piece, (x, y, rotation) in solution.items():

            new_x = x - key_x
            new_y = y - key_y

            adjusted_solution[piece] = np.array([new_x, new_y, rotation])

        return adjusted_solution

    def thread_shutdown(self):
        self.select_anchor_thread.join()
        self.select_neighbour_thread.join()
        self.pl_solver_thread.join()
        # toDo

    def apply_ground_truth(self):
        # debug
        ground_truth = self.path_dic['ground_truth']
        with open(ground_truth, 'r') as f:
            data = json.load(f)
        desired_pieces = [key.replace('.png', '') for key in self.pl_solution.keys()]
        output = {}
        for piece, info in data.items():
            if piece in desired_pieces:
                x, y = info['translation']
                rotation = info['rotation']
                x_adj = int((x + 200) * 2)
                y_adj = int((y + 200) * 2)
                rotation_adj = (rotation + 360) % 360
                output[f'{piece}.png'] = np.array([x_adj, y_adj, rotation_adj])
        return output

    def are_neighbors(self, grabbed_image, image):
        # Extract bounding boxes in [min_x, min_y, max_x, max_y] format
        min_x1, min_y1, max_x1, max_y1 = grabbed_image.extract_bounding_box()
        min_x2, min_y2, max_x2, max_y2 = image.extract_bounding_box()

        # print(image1_bounding_box)
        # print(image2_bounding_box)

        # Check if they overlap or touch
        # Check if the bounding boxes are completely separate
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:  # One is completely to the left/right of the other
            return False
        return True  # No collision

    def setting(self, setting_type):  # unified path setting
        path_dic = ""
        image_path = ""
        mask_path = ""
        comp_path = ""
        pieces_path = ""
        comp_folder = ""
        comp_name = ""
        ground_truth = ""
        dataset_name = ""
        icons_path = "GUI/Icons/"
        apply_gt = False
        number_of_neighbours = 3
        number_of_anchors = 4
        solver_parameters = ""
        setting_dir = ""

        setting_path = os.path.join(setting_dir, setting_type)
        if not os.path.exists(setting_path):
            raise FileNotFoundError(f"Setting path not found: {setting_path}")

        os_path = os.getcwd()

        if os.path.exists(setting_path):
            # with open(setting_path, 'r') as setting_file:
            #     lines = setting_file.readlines()
            #     for line in lines:
            #         # if line.startswith('image_path:'):
            #         #     image_path = os_path + line.split('image_path: ')[1].strip()
            #         # elif line.startswith('mask_path:'):
            #         #     mask_path = os_path + line.split('mask_path: ')[1].strip()
            #         if line.startswith('backend_path:'):
            #             backend_path = os_path + line.split('backend_path: ')[1].strip()
            #         # elif line.startswith('comp_path:'):
            #         #     comp_path = os_path + line.split('comp_path: ')[1].strip()
            #         # elif line.startswith('pieces_path:'):
            #         #     pieces_path = os_path + line.split('pieces_path: ')[1].strip()
            #         # elif line.startswith('comp_folder:'):
            #         #     comp_folder = os_path + line.split('comp_folder: ')[1].strip()
            #         elif line.startswith('comp_name:'):
            #             comp_name = line.split('comp_name: ')[1].strip()
            #         elif line.startswith('Rotation_Intervals:'):
            #             rotation_intervals = line.split('Rotation_Intervals: ')[1].strip()
            #         # elif line.startswith('ground_truth:'):
            #         #     ground_truth = os_path + line.split('ground_truth: ')[1].strip()
            #         elif line.startswith('number_of_neighbours:'):
            #             number_of_neighbours = int(line.split('number_of_neighbours: ')[1].strip())
            #         elif line.startswith('comp_format:'):
            #             comp_format = line.split('comp_format: ')[1].strip()
            #         # elif line.startswith('apply_gt:'):
            #         #     apply_gt = line.split('apply_gt: ')[1].strip()
            #         elif line.startswith('parameters:'):
            #             parameters = os_path + line.split('parameters: ')[1].strip()
            #         elif line.startswith('number_of_anchors:'):
            #             number_of_anchors = int(line.split('number_of_anchors: ')[1].strip())
            #         # elif line.startswith('dataset_name:'):
            #         #     dataset_name = line.split('dataset_name: ')[1].strip()
            #         elif line.startswith('solver_parameters:'):
            #             solver_parameters = os_path + line.split('solver_parameters: ')[1].strip()
            #         elif line.startswith('icons:'):
            #             icons_path = os_path + line.split('icons: ')[1].strip()
            self.yaml, params = self.extract_yaml(setting_path)
            backend_path = self.yaml.data_folder
            print("backend_path", backend_path)
            image_path = self.yaml.get_puzzle_images_subfolder()
            print("image_path", image_path)
            mask_path = self.yaml.get_puzzle_masks_subfolder()
            comp_path = self.yaml.get_CM_path()
            comp_name = os.path.basename(os.path.normpath(comp_path))
            comp_folder = os.path.dirname(os.path.normpath(comp_path))
            pieces_path = image_path
            rotation_intervals = params['compatibility']['grid']['theta_step']
            cm_yaml = self.yaml.get_CM_output_parameters_path()
            parameters = cm_yaml
            comp_format = 'R'
            number_of_pieces = len(self.yaml.get_puzzle_pieces_filenames())
            number_of_anchors = number_of_pieces
            number_of_neighbours = number_of_pieces
            dataset_name = self.yaml.get_puzzle_name()
            solver_parameters = params['solver']
            ground_truth = self.yaml.ground_truth_filename
            apply_gt = False
            # image_path = os_path + backend_path + "images/"
            # mask_path = os_path + backend_path + "binary_masks/"
            # comp_path = os_path + backend_path + "exp/CM_output_params.yaml"
            # pieces_path = os_path + backend_path + "images/"
            # comp_folder = os_path + backend_path + "exp"
            # dataset_name = os.path.basename(os.path.normpath(backend_path))
            # apply_gt = False
            # ground_truth = os_path + backend_path + ""
            cache_path = "/GUI/Cache/"
            self.cache_path = os_path + cache_path
            # sftp = self.get_sftp_client()
            # with self.get_sftp_client() as sftp:
            #     print(sftp.listdir("/home/ssd/datasets/RePAIR_Demo"))
            path_dic = {'image_path': image_path, 'mask_path': mask_path, 'backend_path': backend_path,
                        'comp_path': comp_path,
                        'pieces_path': pieces_path, 'comp_folder': comp_folder, 'comp_name': comp_name,
                        'rotation_intervals': rotation_intervals, 'ground_truth': ground_truth,
                        'number_of_neighbours': number_of_neighbours, 'comp_format': comp_format,
                        'apply_gt': apply_gt, 'parameters': parameters, 'number_of_anchors': number_of_anchors,
                        'dataset_name': dataset_name, 'cache_path': self.cache_path, 'solver_parameters': solver_parameters,
                        'icons': icons_path, 'yaml': self.yaml}


            self.set_path(path_dic)
        else:
            raise FileNotFoundError(f"Setting path not found: {setting_path}")

        return path_dic, rotation_intervals, backend_path

    def extract_yaml(self, backend_path):
        config = yaml_config.Configuration()
        params = config.load(backend_path)
        config.set_puzzle_single_run_random_folder_name(params['exp_name'])
        return config, params

    def calculate_results(self, answer, probability, iteration, bucket):
        q_pos = 0
        rmse_translation = 0
        rmse_rot = 0
        threshold = 0.0
        if probability is None:
            evaluated_answer = answer
        else:
            # Create a filtered copy of `answer` based on `probability`
            evaluated_answer = {
                k: v for k, v in answer.items()
                if probability.get(k, 0) >= threshold
            }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # catch all warnings

            # q_pos, rmse_rot, rmse_translation = self.evaluate(evaluated_answer)

            for warning in w:
                if issubclass(warning.category, RuntimeWarning):
                    pass  # if you have 1 item in the evaluated which means that the probability of other pieces are not high enough, you will get RuntimeWarning

        if probability is None:
            self.logger(f"(last_iteration, Final result) : q_pos {q_pos:.5f}   "f"rmse_rot {rmse_rot:.2f}   rmse_translation {rmse_translation:.2f}")
        else:
            self.logger(f"q_pos {q_pos:.5f}   "f"rmse_rot {rmse_rot:.2f}   rmse_translation {rmse_translation:.2f}")
        self.main_app.last_eval_bucket = bucket

    def logger(self, string):
        iteration = f"iteration {self.get_iteration():3d}"
        timestamp = str(time.time())  # seconds since epoch (as float, converted to string)
        with open(self.cache_path + "solver_log.txt", "a") as f:
            f.write(timestamp + " " + iteration + ": " + string + "\n")
            print(timestamp + " " + iteration + ": " + string + "\n")






