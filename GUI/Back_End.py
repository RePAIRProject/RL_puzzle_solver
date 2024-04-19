from threading import Thread, Event, Lock
import select_anchor_RePAIR
import RL_puzzle_solver.HIL.puzzle_solver as puzzle_solver


select_anchor_running = None
select_neighbour_running = None


select_anchor_thread = Thread()
select_neighbour_thread = Thread()
pl_solver_thread = Thread()

select_anchor_lock = Lock()
select_neighbour_lock = Lock()

anchor_images = []
neighbour_images = []

image_names = []
image_scores = []
image_numbers = []

select_anchor_done = False
select_neighbour_done = False

key_fragment = ""
neighbour_ids = []
input_dict = {}


def select_anchor_thread_function():
    global anchor_images
    global image_names
    global image_scores
    global image_numbers
    set_select_anchor_running(True)
    anchor_images = select_anchor_RePAIR.select_anchor()
    extract_lists(anchor_images)
    set_select_anchor_running(False)
    set_select_anchor_done(True)


def pl_solver_thread_function():
    global input_dict
    i = puzzle_solver.assemble(input_dict)
    print("puzzle solver solution:", i)


def select_neighbour_thread_function():
    global neighbour_images
    global image_names
    global image_scores
    global image_numbers
    set_select_neighbour_running(True)
    neighbour_images = select_anchor_RePAIR.select_neighbour()
    extract_lists(neighbour_images)
    set_select_neighbour_running(False)
    set_select_neighbour_done(True)


def extract_lists(main_list):
    global image_names
    global image_scores
    global image_numbers
    image_names = []
    image_scores = []
    image_numbers = []
    for i in range(len(main_list)):
        image_names.append(main_list[i][0])
        image_scores.append(main_list[i][1])
        image_numbers.append(i)
    for i in range(len(main_list)):
        print(image_numbers[i], ':',  "Path:", image_names[i], "Score:",  image_scores[i])


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


def set_select_neighbour_running(boolean):
    global select_neighbour_running
    select_neighbour_lock.acquire()
    select_neighbour_running = boolean
    select_neighbour_lock.release()


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

    for i in range(len(neighbour_ids)):
        print(i, neighbour_ids[i])

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


def thread_shutdown():
    select_anchor_thread.join()
    # toDo
