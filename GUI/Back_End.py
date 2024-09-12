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
neighbour_ids = []
input_dict = {}
pl_solution = {}

backend_path = ""
path = ""
path_bw = ""

path_dic = None


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


def select_neighbour_thread_function():
    global neighbour_images
    global image_names
    global image_scores
    global image_numbers
    global backend_path
    global path
    global path_bw
    global path_dic
    set_select_neighbour_running(True)
    neighbour_images = select_anchor_RePAIR.select_neighbour(path_dic, key_fragment)
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
    input_dict = {'anchor': key_fragment, 'neighbours': neighbour_ids, 'puzzle': "repair_g28"}

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
        print("pl solution: " + str(pl_solution))
    pl_solution = scale_solution()
    return pl_solution


def scale_solution():
    global pl_solution
    global path_dic
    key_x, key_y, key_rotation = pl_solution[key_fragment]
    parametrs = path_dic['parameters']
    with open(parametrs, 'r') as f:
        data = json.load(f)

    adjusted_solution = {}

    for piece, (x, y, rotation) in pl_solution.items():
        # Adjust the position relative to the key_fragment
        new_x = x - key_x
        new_y = y - key_y
        # Keep the rotation unchanged
        adjusted_solution[piece] = np.array([new_x, new_y, rotation])

    # Show the adjusted result
    print(adjusted_solution)
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
