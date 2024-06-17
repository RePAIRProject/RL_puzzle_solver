import os

from kivy import Config
from kivy.clock import Clock, mainthread
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivymd.app import MDApp
import argparse
import numpy as np
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar

from kivy.core.window import Window

import Back_End
from select_anchor_RePAIR import get_backend_path

import threading
import time

import PuzzlePiece

from MoveableImage import MovableImage

Window.clearcolor = (0, 0, 0, 0)

backend_path = os.getcwd() + "/GUI/Images/RePAIR_plaque_2/"

showed_image_list = []
current_image_list = []
neighbour_ids = []
buttons = []
file_names = []

pl_solution = {}

key_fragment_id = ""

image_is_set = False
anchor_showed = False
neighbour_showed = False
solution_applied = False
initial_image_updates = False
hold_left = False
checked_border = False
test_thread_started = True

communicate_thread_lock = threading.Lock()
click_label = Label()

grabbed_image = None
key_image = None
sorted_image_scores = None

select_anchor_running = None
select_neighbour_running = None
pl_solver_running = None

select_anchor_done = None
select_neighbour_done = None
pl_solver_done = None

selected_pic = 0

ratio = 1
time_stamp = 0
none_counter = 0
keyboard_input = 0
key_offset_x = 0
key_offset_y = 0

toolbar_color = 0  # 0 for light blue, -1 for red, 1 for green

communication_freq = 0.25  # in seconds
graphic_freq = 0.25  # in seconds


class MainLayout(GridLayout):  # might need to change GridLayout to sth else to be fix some bugs (not as important)
    def __init__(self):
        super().__init__()

    the_list = []


class GUIApp(MDApp):
    widget_list = []
    widget_dict = {}
    global select_anchor_running
    global toolbar_color
    toolbar_bg = 0
    global backend_path
    grid_layout = GridLayout()
    is_grabbing_window = False

    def build(self):

        the_app = self
        the_layout = MDBoxLayout(md_bg_color=(0, 0, 0, 1))
        # the_layout = Scatter()
        main_layout = MainLayout()
        main_layout.cols = 1

        Window.bind(on_motion=self.on_touch_move, on_key_down=self._on_keyboard_down, on_key_up=self.on_keyboard_up,
                    on_resize=self.on_resize)

        main_layout.rows = 3

        toolbar = MDTopAppBar()
        toolbar.orientation = "horizontal"

        main_layout.add_widget(toolbar)

        anchor_button = Button(text="Select Anchor")
        anchor_button.bind(on_press=start_select_anchor)
        anchor_button.size_hint_x = 0.5

        show_button = Button(text="Show")
        show_button.bind(on_press=self.show_images)
        show_button.size_hint_x = 0.5

        neighbour_button = Button(text="Neighbour")
        neighbour_button.bind(on_press=start_select_neighbour)
        neighbour_button.size_hint_x = 0.5

        pl_solver_button = Button(text="PL Solver")
        pl_solver_button.bind(on_press=start_pl_solver)
        pl_solver_button.size_hint_x = 0.5

        toolbar.left_action_items.append(["menu", lambda x: the_app.callback()])

        toolbar.add_widget(pl_solver_button)
        toolbar.add_widget(neighbour_button)
        toolbar.add_widget(anchor_button)
        toolbar.add_widget(show_button)

        main_layout.minimum_height = 1

        click_label.text = "Click on the pictures"
        click_label.size_hint_y = 0.1
        click_label.height = 0.1

        grid_layout = GridLayout()
        grid_layout.cols = 5

        main_layout.add_widget(grid_layout)
        main_layout.add_widget(click_label)

        neighbour_button.disabled = True
        the_layout.add_widget(main_layout)
        self.widget_list.append(toolbar)  # 0 toolbar
        self.widget_list.append(anchor_button)  # 1 anchor_button
        self.widget_list.append(show_button)  # 2 show_button
        self.widget_list.append(grid_layout)  # 3 image_view
        self.widget_list.append(neighbour_button)  # 4 neighbour_button

        self.widget_dict.update({'grid_layout': grid_layout})
        self.widget_dict.update({'toolbar': toolbar})
        self.widget_dict.update({'anchor_button': anchor_button})
        self.widget_dict.update({'show_button': show_button})
        self.widget_dict.update({'neighbour_button': neighbour_button})
        self.widget_dict.update({'main_layout': main_layout})

        Clock.schedule_interval(self.checking_clock, graphic_freq)  # Graphic Internal Thread to communicate
        return the_layout

    @mainthread
    def set_images(self, has_score, *args, **kwargs):
        global backend_path
        global file_names
        global current_image_list
        global initial_image_updates
        global key_image

        scores = Back_End.image_scores
        current_image_list = []
        # score_label.text = str(scores[i])
        backend_path = get_backend_path()
        file_names = Back_End.image_names
        if has_score == 0 & (key_image is not None):
            file_names.append(key_image.get_id())
            Back_End.image_numbers += 1
            scores.append(0)
        for i in range(Back_End.image_numbers):

            image = image_reader(i, scores[i], has_score)
            # image.fit_mode = "contain"
            current_image_list.append(image)
        initial_image_updates = False

    @mainthread
    def on_touch_move(self, window, pos, touch, *args, **kwargs):  # Mouse Listener
        global click_label
        global grabbed_image
        global hold_left
        global checked_border
        global none_counter
        global keyboard_input
        global key_fragment_id
        global key_image

        scrolling = 0
        print(keyboard_input)
        if keyboard_input == 304:
            self.is_grabbing_window = True
        if keyboard_input == 305:
            if touch.button == 'scrollup':  # scroll up is scrolling down :|
                if grabbed_image is not None:
                    grabbed_image.rotate(-1)
            elif touch.button == 'scrolldown':  # scrolldown is scrolling up :|
                if grabbed_image is not None:
                    grabbed_image.rotate(1)

        if 'button' in touch.profile:  # may cause bug in different systems -_- /todo
            if not hasattr(touch, 'prev_mouse') or not (touch.prev_mouse == touch.button):
                touch.prev_mouse = touch.button
                hold_left = False
            else:
                if touch.button == 'left':
                    hold_left = True
                else:
                    hold_left = False
            none_counter = 0
        else:
            if none_counter > 1:
                none_counter = 0
                hold_left = False
            else:
                none_counter += 1  # weird input recognition from KIVY
        window_size = Window.size
        mouse_pos = (window_size[0] * touch.spos[0], window_size[1] * touch.spos[1])

        if touch.button == 'right':
            grabbed_image = None

        if touch.button == 'left':
            if not hold_left:
                checked_border = False
                for image in current_image_list:
                    if not hasattr(touch, 'dragging') or not touch.dragging:
                        if image.collides(mouse_pos[0], mouse_pos[1]):
                            if not checked_border:
                                width_height = ((image.width - image.norm_image_size[0]) / 2,
                                                (image.height - image.norm_image_size[1]) / 2)
                                pixel = map_mouse_pos_pixel(image.get_real_pos(), image.texture_size,
                                                            image.get_norm_image_size(), mouse_pos, width_height,
                                                            (image.texture_size[0]/image.norm_image_size[0]),
                                                            (image.texture_size[1]/image.norm_image_size[1]))

                                if image.check_mask(pixel):
                                    grabbed_image = image
                                    grabbed_image.update_translate()
                                    print("grabbed image: ", grabbed_image.texture_size, "here", grabbed_image.norm_image_size)
                                    checked_border = True  # /todo
                                    break
                if not checked_border:
                    grabbed_image = None
            else:
                if grabbed_image is not None and checked_border:
                    if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
                        # Store the offset between touch position and widget position
                        touch.offset_x = mouse_pos[0] - grabbed_image.get_real_pos()[0]
                        touch.offset_y = mouse_pos[1] - grabbed_image.get_real_pos()[1]
                        # grabbed_image.translate(touch.offset_x, touch.offset_y)
                    grabbed_image.translate(mouse_pos[0] - touch.offset_x, mouse_pos[1] - touch.offset_y)

                    if (not select_anchor_running) and (not select_neighbour_running) and (not select_neighbour_done):
                        key_fragment_id = str(grabbed_image.get_id())
                        key_image = grabbed_image
                    click_label.text = str(grabbed_image.get_number() + 1)
                else:
                    if self.is_grabbing_window:  # left shift
                        print("here")
                        grid_layout = self.widget_dict['grid_layout']
                        if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
                            # Store the initial touch position
                            touch.offset_x = mouse_pos[0]
                            touch.offset_y = mouse_pos[1]

                        # Update the position of the layout based on the movement of the mouse
                        grid_layout.pos = (grid_layout.pos[0] + (mouse_pos[0] - touch.offset_x),
                                           grid_layout.pos[1] + (mouse_pos[1] - touch.offset_y))

                        # Update the touch position for the next move event
                        touch.offset_x = mouse_pos[0]
                        touch.offset_y = mouse_pos[1]
                        self.is_grabbing_window = False

    @mainthread
    def on_keyboard_up(self, instance, keyboard, keycode):  # Keyboard up Listener
        global keyboard_input
        if keyboard is not None:
            if keyboard == 305:  # code for ctrl button on keyboard
                keyboard_input = None  # might cause issue
            if keyboard == 304:
                keyboard_input = None

    @mainthread
    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):  # Keyboard down Listener
        global keyboard_input
        keyboard_input = keyboard

    @mainthread
    def on_resize(self, *args):
        if (select_anchor_done is not None) & (select_neighbour_done is not None) & (pl_solver_done is not None):
            if select_anchor_done & select_neighbour_done & pl_solver_done:
                self.apply_solution()

    @mainthread
    def show_images(self, *args, **kwargs):
        global showed_image_list
        global current_image_list
        global time_stamp
        if len(showed_image_list) != 0:
            self.clear_images()

        showed_image_list = []
        time_stamp = time.time()
        for i in range(len(current_image_list)):
            showed_image_list.append(current_image_list[i].get_grid())
            self.widget_list[3].add_widget(showed_image_list[i])
            time_stamp = time.time()

    def clear_images(self, *args, **kwargs):
        global showed_image_list
        for i in range(len(showed_image_list)):
            self.widget_list[3].remove_widget(showed_image_list[i])
        showed_image_list = []

    @mainthread
    def apply_solution(self):
        global pl_solution
        global key_fragment_id
        global key_image
        global key_offset_x
        global key_offset_y
        center_x = (Window.size[0] / 2)  # Calculate the center of the window in x-axis
        center_y = (Window.size[1] / 2)  # Calculate the center of the window in y-axis

        print("center_x", center_x, "center_y", center_y)

        for image in current_image_list:
            image_id = image.get_id()

            # Check if the image ID exists in pl_solution
            if image_id in pl_solution:
                positions = pl_solution[image_id]

                # If the image is the key fragment, calculate its offset from the center
                if image_id == key_fragment_id:
                    position = np.array([positions[1], -1 * positions[0]])  # fix the coordinates
                    key_offset_x = center_x - position[0]
                    key_offset_y = center_y - position[1]
                    break

        print("key_offset_x", key_offset_x, "key_offset_y", key_offset_y)

        for image in current_image_list:
            image_id = image.get_id()

            if image_id in pl_solution:
                positions = pl_solution[image_id]

                position = np.array([positions[1], -1 * positions[0]])  # fix the coordinates

                new_positions = np.array([position[0] + key_offset_x, position[1] + key_offset_y])
                print("image_id", image_id, "positions", new_positions)

                r = np.array([image.texture_size[0] / image.norm_image_size[0],
                              image.texture_size[1] / image.norm_image_size[1]])
                image.update(new_positions, r)

    def checking_clock(self, *args, **kwargs):
        global selected_pic
        global anchor_showed
        global neighbour_showed
        global initial_image_updates
        global solution_applied
        global pl_solution
        communicate_thread_lock.acquire()
        self.toolbar_changes(toolbar_color)

        clicked = click_label.text
        if (clicked.isdigit()) & (selected_pic == 0):
            self.widget_list[4].disabled = False
            selected_pic = int(clicked)
        if clicked.isdigit():
            selected_pic = int(clicked)
        if not anchor_showed:
            if (Back_End.get_select_anchor_done()) & (len(current_image_list) == 0):
                self.set_images(1)
                self.show_images(self)
                anchor_showed = True
        elif ((Back_End.get_select_anchor_done()) & (len(current_image_list) == 0) &
              (Back_End.get_select_neighbour_done()) & (not neighbour_showed)):
            self.set_images(0)
            self.show_images(self)
            neighbour_showed = True
        elif ((Back_End.get_select_anchor_done()) & (Back_End.get_select_neighbour_done()) & neighbour_showed &
              Back_End.get_pl_solver_done() & (not solution_applied)):
            pl_solution = Back_End.get_pl_solution()
            self.apply_solution()
            solution_applied = True
        if (time.time() - time_stamp > 1) and not initial_image_updates:
            for i in range(len(current_image_list)):
                current_image_list[i].update_virtual_pos()
            initial_image_updates = True
        communicate_thread_lock.release()

    def toolbar_changes(self, color):
        match color:
            case 0:
                if self.toolbar_bg != 0:
                    self.widget_list[0].md_bg_color = (0.678431373, 0.847058824, 0.901960784, 1)  # Set Toolbar Blue
                    self.widget_list[2].disabled = True
                    self.toolbar_bg = 0
            case -1:
                if self.toolbar_bg != -1:
                    self.widget_list[0].md_bg_color = (0.545098039, 0, 0, 1)  # Set Toolbar Red
                    self.widget_list[1].disabled = True
                    self.widget_list[2].disabled = True
                    self.toolbar_bg = -1
            case 1:
                if self.toolbar_bg != 1:
                    self.widget_list[0].md_bg_color = (0.141176471, 0.529411765, 0.129411765, 1)  # Set Toolbar Green
                    self.widget_list[1].disabled = True
                    self.widget_list[2].disabled = False  # todo find sth else for this...
                    self.toolbar_bg = 1

    def callback(self):
        # start_select_anchor(self)
        return


def start_select_anchor(self):
    global image_is_set
    image_is_set = False
    Back_End.start_anchor_thread()


def start_select_neighbour(self):
    global current_image_list
    global image_is_set
    global key_fragment_id

    Back_End.key_fragment = key_fragment_id
    image_is_set = False
    current_image_list = []
    Back_End.start_neighbour_thread()


def start_pl_solver(self):
    global current_image_list
    global image_is_set
    global neighbour_ids

    for i in range(0, len(current_image_list)):
        if not (current_image_list[i].get_id() == key_image.get_id()):
            neighbour_ids.append(current_image_list[i].get_id())
    Back_End.neighbour_ids = neighbour_ids
    # image_is_set = False
    # current_image_list = []
    Back_End.start_pl_solver_thread()


def communicate_thread():  # communication thread, to communicate between UI, Graphic and BackEnd
    global select_anchor_running
    global select_neighbour_running
    global pl_solver_running
    global select_anchor_done
    global select_neighbour_done
    global pl_solver_done
    global toolbar_color
    while True:
        communicate_thread_lock.acquire()

        select_anchor_running = Back_End.get_select_anchor_running()
        select_neighbour_running = Back_End.get_select_neighbour_running()
        pl_solver_running = Back_End.get_pl_solver_running()
        select_neighbour_done = Back_End.get_select_neighbour_done()
        select_anchor_done = Back_End.get_select_anchor_done()
        pl_solver_done = Back_End.get_pl_solver_done()

        if select_anchor_running is not None:
            if select_anchor_running:
                toolbar_color = -1
            else:
                toolbar_color = 1
        if select_anchor_done:
            toolbar_color = 1
        if select_neighbour_running is not None:
            if select_neighbour_running:
                toolbar_color = -1
            else:
                toolbar_color = 1
        else:
            toolbar_color = 0
        communicate_thread_lock.release()
        time.sleep(communication_freq)  # Thread sleep timerfasd


def map_mouse_pos_pixel(image_pos, image_pixel, image_size, mouse_pos, width_height, ratio_x, ratio_y):
    global ratio

    ratio = (ratio_x, ratio_y)
    print(ratio)

    relative_pos = (mouse_pos[0] - image_pos[0] - width_height[0], mouse_pos[1] - image_pos[1] - width_height[1])
    reality_pixel = (relative_pos[0] * ratio[0], relative_pos[1] * ratio[1])
    return reality_pixel


def get_args():
    parser = argparse.ArgumentParser(description='Select anchor / key fragment')

    parser.add_argument('-d', '--dataset', type=str,
                        default=backend_path + 'RGBA_merged',
                        help='data folder')

    parser.add_argument('-f', '--fg_mask', type=str,
                        default=backend_path + 'FG_merged',
                        help='data folder')
    answer = parser.parse_args()
    return answer


def image_reader(image_number, score, has_score):
    path = file_names[image_number]

    source = backend_path + "RGBA_merged/" + path

    click_label.color = (1, 0, 1, 1)
    image = MovableImage(source, click_label, score, image_number, has_score, path, 0)

    return image


if __name__ == '__main__':
    Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
    test_thread = threading.Thread(target=communicate_thread, daemon=True)
    test_thread_started = True
    test_thread.start()

    app = GUIApp().run()
