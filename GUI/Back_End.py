from threading import Thread, Event, Lock
import select_anchor_RePAIR


select_anchor_running = None
select_anchor_thread = Thread()

select_anchor_lock = Lock()
anchor_images = []


def select_anchor_thread_function():
    global select_anchor_running
    global anchor_images
    set_select_anchor_running(True)
    anchor_images = select_anchor_RePAIR.select_anchor()
    set_select_anchor_running(False)


def set_select_anchor_running(boolean):
    global select_anchor_running
    select_anchor_lock.acquire()
    select_anchor_running = boolean
    select_anchor_lock.release()


def start_back_end():
    global select_anchor_thread
    select_anchor_thread = Thread(target=select_anchor_thread_function, daemon=True)
    select_anchor_thread.start()


def get_select_anchor_running():
    select_anchor_lock.acquire()
    answer = select_anchor_running
    select_anchor_lock.release()
    return answer


def thread_shutdown():
    select_anchor_thread.join()
    # toDo
