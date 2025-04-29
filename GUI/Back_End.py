from threading import Thread, Event, Lock

from scipy.io import loadmat
from scipy.spatial import KDTree


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import select_anchor_RePAIR as select_anchor_RePAIR
import RL_puzzle_solver.HIL.puzzle_solver as puzzle_solver
import json
import numpy as np
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

        self.mat = None
        self.R = None

    def set_path(self, path_dic):
        self.path_dic = path_dic

        self.backend_path = self.path_dic['backend_path']
        self.path = self.path_dic['image_path']
        self.path_bw = self.path_dic['mask_path']

        comp_folder = path_dic['comp_folder']
        comp_name = path_dic['comp_name']

        self.mat = loadmat(os.path.join(comp_folder, comp_name))
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

        puzzle_solver.set_cm_element(main, neighbour, relative_position, value)

    def set_p_elements(self, couple, offset):
        image_name = couple[0]
        image_pos = couple[1]
        pos = image_pos

        xy_step, theta_step = self.extract_steps()

        # pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
        pos = self.reverse_offset(pos, offset)

        pos = self.scale_to_solver(xy_step, theta_step, pos, self.path_dic)

        puzzle_solver.set_p_elements(pos[0], pos[1], pos[2], image_name)


    def reverse_offset(self, pos, offset):
        pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
        return pos

    def extract_steps(self):
        parameters = self.path_dic['parameters']
        xy_step = 1
        theta_step = 360
        with open(parameters, 'r') as f:
            data = json.load(f)
            if data is not None:
                xy_step = data['xy_step']
                theta_step = data['theta_step']
        return xy_step, theta_step

    def get_solution_dict(self):
        answer, probability, process, iteration = puzzle_solver.get_solution_dict()

        if answer is not None:
            answer = self.scale_solution(answer)
        return answer, probability, process

    def get_API_solution(self):
        answer, probability, process, iteration = puzzle_solver.get_solution_dict()
        # answer = self.scale_solution(answer)
        return answer, probability, process, iteration

    def solver_toggle_lock(self, value):
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
        mat = loadmat(os.path.join(comp_folder, comp_name))  # load the new compatibility matrix

        R = mat[dic['comp_format']]

        bias = R.shape[0] + 1

        position = [x + bias, y + bias, rotation]
        return position

    def kill_puzzle_solver(self):
        if self.pl_solver_running:
            puzzle_solver.solver_alive(False)

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
        self.pl_solution = self.scale_solution(self.pl_solution)
        return self.pl_solution

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
        cache_path = ""
        icons_path = ""
        apply_gt = False
        number_of_neighbours = 3
        number_of_anchors = 4
        solver_parameters = ""
        setting_dir = ""
        print("os_path", os.path)
        setting_path = os.path.join(setting_dir, setting_type)
        print("setting_path", setting_path)
        if not os.path.exists(setting_path):
            with open(setting_path, "w") as setting_file:
                setting_file.writelines(["image_path: /GUI/DataBase/Images/RePAIR_plaque_2/RGBA_merged/",
                                         "\n",
                                         "mask_path: /GUI/DataBase/Images/RePAIR_plaque_2/FG_merged/",
                                         "\n",
                                         "backend_path: /GUI/DataBase/Images/RePAIR_plaque_2/",
                                         "\n",
                                         "comp_path: /GUI/DataBase/output/repair_g28/compatibility_parameters.json",
                                         "\n",
                                         "pieces_path: /GUI/DataBase/output/repair_g28/pieces/",
                                         "\n",
                                         "comp_folder: /GUI/DataBase/output/repair_g28/compatibility_matrix/",
                                         "\n",
                                         "comp_name: CM_linesdet_manual_cost_LAP.mat",
                                         "\n",
                                         "Rotation_Intervals: 1",
                                         "\n",
                                         "number_of_neighbours: 3",
                                         "\n",
                                         "comp_format: R_line",
                                         "\n",
                                         "apply_gt: False",
                                         "\n",
                                         "parameters: /GUI/DataBase/output/repair_g28/compatibility_parameters.json",
                                         "\n",
                                         "number_of_anchors: 4",
                                         "\n",
                                         "dataset_name: RePair_group_28",
                                         "\n",
                                         "icons: /GUI/Icons/",
                                         "\n"
                                         ])
        os_path = os.getcwd()
        print("os_path", os_path)
        if os.path.exists(setting_path):
            with open(setting_path, 'r') as setting_file:
                lines = setting_file.readlines()
                for line in lines:
                    if line.startswith('image_path:'):
                        image_path = os_path + line.split('image_path: ')[1].strip()
                    elif line.startswith('mask_path:'):
                        mask_path = os_path + line.split('mask_path: ')[1].strip()
                    elif line.startswith('backend_path:'):
                        backend_path = os_path + line.split('backend_path: ')[1].strip()
                    elif line.startswith('comp_path:'):
                        comp_path = os_path + line.split('comp_path: ')[1].strip()
                    elif line.startswith('pieces_path:'):
                        pieces_path = os_path + line.split('pieces_path: ')[1].strip()
                    elif line.startswith('comp_folder:'):
                        comp_folder = os_path + line.split('comp_folder: ')[1].strip()
                    elif line.startswith('comp_name:'):
                        comp_name = line.split('comp_name: ')[1].strip()
                    elif line.startswith('Rotation_Intervals:'):
                        rotation_intervals = line.split('Rotation_Intervals: ')[1].strip()
                    elif line.startswith('ground_truth:'):
                        ground_truth = os_path + line.split('ground_truth: ')[1].strip()
                    elif line.startswith('number_of_neighbours:'):
                        number_of_neighbours = int(line.split('number_of_neighbours: ')[1].strip())
                    elif line.startswith('comp_format:'):
                        comp_format = line.split('comp_format: ')[1].strip()
                    elif line.startswith('apply_gt:'):
                        apply_gt = line.split('apply_gt: ')[1].strip()
                    elif line.startswith('parameters:'):
                        parameters = os_path + line.split('parameters: ')[1].strip()
                    elif line.startswith('number_of_anchors:'):
                        number_of_anchors = int(line.split('number_of_anchors: ')[1].strip())
                    elif line.startswith('dataset_name:'):
                        dataset_name = line.split('dataset_name: ')[1].strip()
                    elif line.startswith('solver_parameters:'):
                        solver_parameters = os_path + line.split('solver_parameters: ')[1].strip()
                    elif line.startswith('icons:'):
                        icons_path = os_path + line.split('icons: ')[1].strip()
        cache_path = "/GUI/pieces/"
        cache_path = os_path + cache_path
        path_dic = {'image_path': image_path, 'mask_path': mask_path, 'backend_path': backend_path,
                    'comp_path': comp_path,
                    'pieces_path': pieces_path, 'comp_folder': comp_folder, 'comp_name': comp_name,
                    'rotation_intervals': rotation_intervals, 'ground_truth': ground_truth,
                    'number_of_neighbours': number_of_neighbours, 'comp_format': comp_format,
                    'apply_gt': apply_gt, 'parameters': parameters, 'number_of_anchors': number_of_anchors,
                    'dataset_name': dataset_name, 'cache_path': cache_path, 'solver_parameters': solver_parameters,
                    'icons': icons_path}

        self.set_path(path_dic)

        return path_dic, rotation_intervals, backend_path






