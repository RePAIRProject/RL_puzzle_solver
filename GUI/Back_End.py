from threading import Thread, Event, Lock
import select_anchor_RePAIR
import RL_puzzle_solver.HIL.puzzle_solver as puzzle_solver
import json
import numpy as np


select_anchor_running = None
select_neighbour_running = None
pl_solver_running = None


select_anchor_thread = Thread()
select_neighbour_thread = Thread()
pl_solver_thread = Thread()

select_anchor_lock = Lock()
select_neighbour_lock = Lock()
pl_solver_lock = Lock()

anchor_images = []
neighbour_images = []

image_names = []
image_scores = []
image_numbers = 0

select_anchor_done = False
select_neighbour_done = False
pl_solver_done = False

key_fragment = ""
solved_pieces = []
neighbour_ids = []
input_dict = {}
pl_solution = {}

backend_path = ""
path = ""
path_bw = ""

path_dic = None
sorted_neighbour_images = None
neighbour_after = 0


def set_backend_path(dic):
    global backend_path
    global path
    global path_bw
    global path_dic
    backend_path = dic['backend_path']
    path = dic['image_path']
    path_bw = dic['mask_path']
    path_dic = dic


def select_anchor_thread_function():
    global anchor_images
    global image_names
    global image_scores
    global image_numbers
    global backend_path
    global path
    global path_bw
    global path_dic
    set_select_anchor_running(True)
    anchor_images = select_anchor_RePAIR.select_anchor(path_dic)
    extract_lists(anchor_images)
    set_select_anchor_running(False)
    set_select_anchor_done(True)


def pl_solver_thread_function():
    global input_dict
    global pl_solution
    global path_dic
    set_pl_solver_running(True)
    pl_solution = puzzle_solver.assemble(input_dict, path_dic)
    set_pl_solver_running(False)
    set_pl_solver_done(True)


def get_next_neighbour(image_id):
    global path_dic
    global sorted_neighbour_images
    global neighbour_after
    boolean = False
    next_neighbour = None
    neighbour_number = path_dic['number_of_neighbours'] + neighbour_after
    if sorted_neighbour_images is not None:
        if neighbour_number >= len(sorted_neighbour_images):
            neighbour_after = -1 * path_dic['number_of_neighbours']
        neighbour_number = path_dic['number_of_neighbours'] + neighbour_after
        next_neighbour = sorted_neighbour_images[neighbour_number]
    for image in neighbour_images:
        if image_id == image[0]:
            neighbour_images.remove(image)
            neighbour_images.append(next_neighbour)
            neighbour_after += 1
            extract_lists(neighbour_images)
            boolean = True
    return boolean, neighbour_images


def select_neighbour_thread_function():
    global neighbour_images
    global image_names
    global image_scores
    global image_numbers
    global backend_path
    global path
    global path_bw
    global path_dic
    global sorted_neighbour_images
    set_select_neighbour_running(True)

    sorted_neighbour_images = select_anchor_RePAIR.select_neighbour(path_dic, key_fragment)

    neighbour_numbers = path_dic['number_of_neighbours']
    neighbour_images = sorted_neighbour_images[:neighbour_numbers]

    extract_lists(neighbour_images)
    set_select_neighbour_running(False)
    set_select_neighbour_done(True)


def extract_lists(main_list):
    global image_names
    global image_scores
    global image_numbers
    image_names = []
    image_scores = []
    image_numbers = 0
    for i in range(len(main_list)):
        image_names.append(main_list[i][0])
        image_scores.append(main_list[i][1])
        image_numbers += 1


def loop_finalization(solved_list, offset, center):
    parameters = path_dic['parameters']
    print("parameters")
    data = None
    xy_step = 1
    theta_step = 360
    with open(parameters, 'r') as f:
        data = json.load(f)
        if data is not None:
            xy_step = data['xy_step']
            theta_step = data['theta_step']
            print(data['p_hs'])
            print(data.keys())


    for pieces in solved_list:
        name = pieces[0]
        pos = pieces[1]
        print(pos)

        pos = [pos[0] - offset[0], pos[1] - offset[1], pos[2]]
        print(pos)

        pos = scale_to_solver(xy_step, theta_step, pos)
        print(name, pos)
        solved_pieces.append(pieces)


def scale_to_solver(xy_step, theta_step, position):
    y, x, rotation = position
    x = -1 * x / xy_step
    y = y / xy_step
    x = round(x) # should I
    y = round(y) # should I?
    rotation = rotation / theta_step

    # centralizing in solution how can I get 16 from?! #ask LUCA
    position = [x + 16, y + 16, rotation]
    return position


def start_anchor_thread():
    global select_anchor_thread
    select_anchor_thread = Thread(target=select_anchor_thread_function, daemon=True)
    select_anchor_thread.start()


def start_neighbour_thread():
    global select_neighbour_thread
    select_neighbour_thread = Thread(target=select_neighbour_thread_function, daemon=True)
    select_neighbour_thread.start()


def start_pl_solver_thread():
    global pl_solver_thread
    global input_dict

    input_dict = {'anchor': key_fragment, 'neighbours': neighbour_ids, 'solved_pieces': solved_pieces, 'puzzle': "repair_g28"}

    select_pl_solver = Thread(target=pl_solver_thread_function, daemon=True)
    select_pl_solver.start()


def get_select_anchor_running():
    select_anchor_lock.acquire()
    answer = select_anchor_running
    select_anchor_lock.release()
    return answer


def get_select_neighbour_done():
    select_neighbour_lock.acquire()
    answer = select_neighbour_done
    select_neighbour_lock.release()
    return answer


def get_select_neighbour_running():
    select_neighbour_lock.acquire()
    answer = select_anchor_running
    select_neighbour_lock.release()
    return answer


def get_pl_solver_running():
    pl_solver_lock.acquire()
    answer = pl_solver_running
    pl_solver_lock.release()
    return answer


def get_pl_solver_done():
    pl_solver_lock.acquire()
    answer = pl_solver_done
    pl_solver_lock.release()
    return answer


def set_select_neighbour_done(boolean):
    global select_neighbour_done
    select_neighbour_lock.acquire()
    select_neighbour_done = boolean
    select_neighbour_lock.release()


def set_select_anchor_running(boolean):
    global select_anchor_running
    select_anchor_lock.acquire()
    select_anchor_running = boolean
    select_anchor_lock.release()


def set_select_anchor_done(boolean):
    global select_anchor_done
    select_anchor_lock.acquire()
    select_anchor_done = boolean
    select_anchor_lock.release()


def set_pl_solver_done(boolean):
    global pl_solver_done
    pl_solver_lock.acquire()
    pl_solver_done = boolean
    pl_solver_lock.release()


def set_select_neighbour_running(boolean):
    global select_neighbour_running
    select_neighbour_lock.acquire()
    select_neighbour_running = boolean
    select_neighbour_lock.release()


def set_pl_solver_running(boolean):
    global pl_solver_running
    pl_solver_lock.acquire()
    pl_solver_running = boolean
    pl_solver_lock.release()


def get_select_anchor_done():
    select_anchor_lock.acquire()
    answer = select_anchor_done
    select_anchor_lock.release()
    return answer


def get_pl_solution():
    global pl_solution
    global path_dic
    apply_gt = path_dic['apply_gt']
    if apply_gt == "True":
        pl_solution = apply_ground_truth()
    pl_solution = scale_solution()
    return pl_solution


def scale_solution():
    global pl_solution
    global path_dic
    key_x, key_y, key_rotation = pl_solution[key_fragment]

    adjusted_solution = {}

    for piece, (x, y, rotation) in pl_solution.items():

        new_x = x - key_x
        new_y = y - key_y

        adjusted_solution[piece] = np.array([new_x, new_y, rotation])

    return adjusted_solution


def thread_shutdown():
    select_anchor_thread.join()
    select_neighbour_thread.join()
    pl_solver_thread.join()
    # toDo


def apply_ground_truth():
    # debug
    global pl_solution
    ground_truth = path_dic['ground_truth']
    with open(ground_truth, 'r') as f:
        data = json.load(f)
    desired_pieces = [key.replace('.png', '') for key in pl_solution.keys()]
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


def create_meta_fragment():

    pass
    # toDo
