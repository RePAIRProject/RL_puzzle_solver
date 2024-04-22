import os

from kivy import Config
from kivy.clock import Clock, mainthread
from kivymd.app import MDApp
import cv2
import argparse
import numpy as np
from kivy.graphics import Rotate, PopMatrix, PushMatrix
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar

from shapely.geometry import Point, Polygon
from point_Inside import is_inside_sm

from kivy.core.window import Window

import RL_puzzle_solver.HIL.puzzle_solver as puzzle_solver

import Back_End
from select_anchor_RePAIR import get_backend_path

import threading
import time

from MoveableImage import MovableImage

# class GUI(Widget):
#     pass
# image_numbers = []
file_names = []
# scores = []
Window.clearcolor = (0, 0, 0, 0)

# Window.fullscreen = 'auto'


# Window.borderless = True
# Window.background_color = (1, 1, 1)

# backEnd_path = "Images/RePAIR_plaque_2/top10/"
backend_path = os.getcwd() + "/GUI/Images/RePAIR_plaque_2/"
showed_image_list = []
current_image_list = []
neighbour_ids = []

pl_solution = {}

none_counter = 0
keyboard_input = 0

key_fragment_id = ""


def image_reader(image_number, score, has_score):
    path = file_names[image_number]
    # print(file_names)
    source = backend_path + "RGBA_merged/" + path

    click_label.color = (1, 0, 1, 1)
    image = MovableImage(source, click_label, score, image_number, has_score, path, 0)
    # image.source = source
    return image


buttons = []


class Image(Image):
    def __init__(self):
        super().__init__()
        self.image_number = None

    # def on_touch_down(self, touch):
    #     if self.collide_point(*touch.pos):
    #         click_label.text = str(self.image_number + 1)


click_label = Label()
image_is_set = False
anchor_showed = False
neighbour_showed = False
solution_applied = False
initial_image_updates = False
ratio = 1

hold_left = False
checked_border = False
time_stamp = 0

grabbed_image = None
key_image = None

communication_freq = 0.25  # in seconds
graphic_freq = 0.25  # in seconds

selected_pic = 0


class MainLayout(GridLayout):
    def __init__(self):
        super().__init__()

    the_list = []


class GUIApp(MDApp):
    widget_list = []

    def build(self):

        # graphic_thread = threading.Thread(target=self.graphic_thread, daemon=True)
        # graphic_thread.start()

        the_app = self
        the_layout = MDBoxLayout(md_bg_color=(0, 0, 0, 1))
        main_layout = MainLayout()
        main_layout.cols = 1

        Window.bind(on_motion=self.on_touch_move, on_key_down=self._on_keyboard_down, on_key_up=self.on_keyboard_up)

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

        # show_button.bind(on_press=self)

        # toolbar_layout = GridLayout()
        # toolbar_layout.cols = 1
        # toolbar_layout.rows = 2
        # toolbar_layout.add_widget(anchor_button)
        # toolbar_layout.add_widget(show_button)
        # toolbar.add_widget(toolbar_layout)

        toolbar.left_action_items.append(["menu", lambda x: the_app.callback()])

        toolbar.add_widget(pl_solver_button)
        toolbar.add_widget(neighbour_button)
        toolbar.add_widget(anchor_button)
        toolbar.add_widget(show_button)

        # main_layout.size_hint = (1, 1)
        main_layout.minimum_height = 1

        click_label.text = "Click on the pictures"
        click_label.size_hint_y = 0.1
        click_label.height = 0.1

        grid_layout = GridLayout()
        grid_layout.cols = 5

        main_layout.add_widget(grid_layout)
        main_layout.add_widget(click_label)
        # grid_layout.col_force_default = 10000
        # widget = Widget()
        # widget.add_widget(grid_layout)

        # grid_layout.add_widget(click_label)

        # for i in range(len(image_numbers)):
        #     score_label = Label()
        #     buttons.append(score_label)
        #     score_label.color = (1, 0, 0, 1)
        #     score_label.text = str(scores[i])
        #
        #     image = image_reader(i, scores[i])
        #
        #     grid_layout.add_widget(image.get_grid())

        # for i in range(10):
        #     button = Button()
        #     button.text = str(i + 1)
        #     button.height = 1000
        #     button.width = 1000
        #
        #     grid_layout.add_widget(button)
        # grid_layout.add_widget(button)
        # grid_layout.add_widget(widget)
        neighbour_button.disabled = True
        the_layout.add_widget(main_layout)
        self.widget_list.append(toolbar)  # 0 toolbar
        self.widget_list.append(anchor_button)  # 1 anchor_button
        self.widget_list.append(show_button)  # 2 show_button
        self.widget_list.append(grid_layout)  # 3 image_view
        self.widget_list.append(neighbour_button)  # 4 neighbour_button

        Clock.schedule_interval(self.checking_clock, graphic_freq)  # Graphic Internal Thread to communicate
        return the_layout

    global select_anchor_running
    global toolbar_color
    toolbar_bg = 0
    global backend_path

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
            # print(i)
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
                for i in range(len(current_image_list)):
                    if not hasattr(touch, 'dragging') or not touch.dragging:
                        if current_image_list[i].collide_point(mouse_pos[0], mouse_pos[1]):
                            if not checked_border:
                                width_height = (
                                    (current_image_list[i].width - current_image_list[i].norm_image_size[0]) / 2,
                                    (current_image_list[i].height - current_image_list[i].norm_image_size[1]) / 2)
                                pixel = map_mouse_pos_pixel(current_image_list[i].pos,
                                                            current_image_list[i].texture_size,
                                                            current_image_list[i].get_norm_image_size(), mouse_pos,
                                                            width_height)

                                if current_image_list[i].check_mask(pixel):
                                    grabbed_image = current_image_list[i]
                                    checked_border = True
                                    break
                if not checked_border:
                    grabbed_image = None
            else:
                if grabbed_image is not None and checked_border:
                    if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
                        # Store the offset between touch position and widget position
                        touch.offset_x = mouse_pos[0] - grabbed_image.x
                        touch.offset_y = mouse_pos[1] - grabbed_image.y
                        # grabbed_image.translate(touch.offset_x, touch.offset_y)
                    grabbed_image.translate(mouse_pos[0] - touch.offset_x, mouse_pos[1] - touch.offset_y)

                    # grabbed_image.x = mouse_pos[0] - touch.offset_x
                    # grabbed_image.y = mouse_pos[1] - touch.offset_y
                    if (not select_anchor_running) and (not select_neighbour_running) and (not select_neighbour_done):
                        key_fragment_id = str(grabbed_image.get_id())
                        key_image = grabbed_image
                    click_label.text = str(grabbed_image.get_number() + 1)

            #     for i in range(len(current_image_list)):
            #         if not hasattr(touch, 'dragging') or not touch.dragging:
            #             if current_image_list[i].collide_point(mouse_pos[0], mouse_pos[1]):
            #                 grabbed_image = current_image_list[i]
            #                 if not hasattr(touch, 'checked_border') or not touch.checked_border:
            #                     width_height = ((grabbed_image.width - grabbed_image.norm_image_size[0]) / 2,
            #                                     (grabbed_image.height - grabbed_image.norm_image_size[1]) / 2)
            #                     pixel = map_mouse_pos_pixel(grabbed_image.pos, grabbed_image.texture_size,
            #                                                 grabbed_image.get_norm_image_size(), mouse_pos, width_height)
            #
            #                     if check_border(pixel, grabbed_image.get_path()):
            #                         touch.checked_border = True
            #                     else:
            #                         # Skip dragging if border check fails
            #                         break
            #         touch.dragging = True
            #
            #         if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
            #             # Store the offset between touch position and widget position
            #             touch.offset_x = mouse_pos[0] - grabbed_image.x
            #             touch.offset_y = mouse_pos[1] - grabbed_image.y
            # else:
            #     if not (grabbed_image is None):
            #         grabbed_image.x = mouse_pos[0] - touch.offset_x
            #         grabbed_image.y = mouse_pos[1] - touch.offset_y
            #         click_label.text = str(grabbed_image.get_number + 1)

    @mainthread
    def on_keyboard_up(self, instance, keyboard, keycode):  # Keyboard up Listener
        global keyboard_input
        if keyboard is not None:
            if keyboard == 305:  # code for ctrl button on keyboard
                keyboard_input = None  # might cause issue

    @mainthread
    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):  # Keyboard down Listener
        global keyboard_input
        keyboard_input = keyboard

    @mainthread
    def show_images(self, *args, **kwargs):
        global showed_image_list
        global current_image_list
        global time_stamp
        if len(showed_image_list) != 0:
            self.clear_images()
        # current_path = os.getcwd() + "/GUI/"

        # path = current_path + "top_10_fragments.json"
        # json_reader(path)
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
        # for i in range(len(current_image_list)):
        #     print(current_image_list[i].get_id())
        # print("puzzle solver solution:", pl_solution)
        # fragments = list(pl_solution.keys())
        # for j in range(len(fragments)):
        #     position = pl_solution[fragments[j]]
        #     print("Puzzle solver", j, ":", fragments[j], "|   Value: ", position)
        #     for k in range(len(position)):
        #         print(position[k])
        # current_image_list.update(position)
        # Iterate over each image in current_image_list
        for image in current_image_list:
            image_id = image.get_id()
            print(image_id)

            # Check if the image ID exists in pl_solution
            if image_id in pl_solution:
                positions = pl_solution[image_id]
                print("Positions for image", image_id, ":", positions)

                # Update the image's position
                image.update(positions, ratio)  # Assuming there's a method update_position for images
            else:
                print("No positions found for image", image_id)

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


sorted_image_scores = None

test_thread_started = True

communicate_thread_lock = threading.Lock()


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


select_anchor_running = None
select_neighbour_running = None
pl_solver_running = None

select_anchor_done = None
select_neighbour_done = None
pl_solver_done = None

toolbar_color = 0  # 0 for light blue, -1 for red, 1 for green


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


def map_mouse_pos_pixel(image_pos, image_pixel, image_size, mouse_pos, width_height):
    global ratio
    ratio = (image_pixel[0] / image_size[0], image_pixel[1] / image_size[1])

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


if __name__ == '__main__':
    Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
    test_thread = threading.Thread(target=communicate_thread, daemon=True)
    test_thread_started = True
    test_thread.start()
    # sorted_image_scores = select_anchor()
    # path = backEnd_path + "top_10_fragments.json"
    # current_path = os.getcwd() + "/GUI/"
    # backEnd_path = get_backend_path()
    # path = current_path + "top_10_fragments.json"
    # json_reader(path)

    app = GUIApp().run()
