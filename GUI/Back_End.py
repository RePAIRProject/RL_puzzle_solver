from threading import Thread, Event, Lock
import select_anchor_RePAIR


select_anchor_running = None
select_neighbour_running = None


select_anchor_thread = Thread()
select_neighbour_thread = Thread()

select_anchor_lock = Lock()
select_neighbour_lock = Lock()

anchor_images = []
neighbour_images = []

image_names = []
image_scores = []
image_numbers = []


def select_anchor_thread_function():
    global select_anchor_running
    global anchor_images
    global image_names
    global image_scores
    global image_numbers
    set_select_anchor_running(True)
    anchor_images = select_anchor_RePAIR.select_anchor()
    for i in range(len(anchor_images)):
        image_names.append(anchor_images[i][0])
        image_scores.append(anchor_images[i][1])
        image_numbers.append(i)
    for i in range(len(anchor_images)):
        print(image_numbers[i], ':',  "Path:", image_names[i], "Score:",  image_scores[i])
    set_select_anchor_running(False)


def select_neighbour_thread_function():
    global select_neighbour_running
    global neighbour_images


def set_select_anchor_running(boolean):
    global select_anchor_running
    select_anchor_lock.acquire()
    select_anchor_running = boolean
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


def get_select_anchor_running():
    select_anchor_lock.acquire()
    answer = select_anchor_running
    select_anchor_lock.release()
    return answer


def thread_shutdown():
    select_anchor_thread.join()
    # toDo
