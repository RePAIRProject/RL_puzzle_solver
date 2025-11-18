import os
import re
from kivy import Config
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
# Config.set('graphics', 'resizable', False)
from kivy.clock import Clock, mainthread
from kivy.metrics import dp
from kivy.uix.image import Image
from kivy.uix.togglebutton import ToggleButton
from kivymd.app import MDApp
import numpy as np
import math
from screeninfo import get_monitors
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Scale
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.card import MDCard
from kivy.core.window import Window
from kivy.uix.progressbar import ProgressBar
from PIL import Image as PILImage
from Back_End import BackEnd
from RL_puzzle_solver.puzzle_utils.puzzle_gen.generator import run_erode
import threading
import time
import shutil
import json
import cv2
from MoveableImage import MovableImage
from MoveableImage import Status
from SandBox import SandBox

Window.clearcolor = (0, 0, 0, 0)
# Window.borderless = True
Window.maximize()

backend_path = os.getcwd() + "/GUI/DataBase/Images/RePAIR_plaque_2/"
path_dic = {}

back_end = BackEnd()

rotation_interval = 0.5
communication_freq = 0.1  # in seconds
graphic_freq = 0.1  # in seconds

class MainLayout(GridLayout):  # might need to change GridLayout to sth else to be fix some bugs (not as important)
    def __init__(self):
        super().__init__()

    the_list = []

class GUIApp(MDApp):
    def __init__(self):
        super().__init__()
        self.the_app = self
        self.communicate_thread_lock = threading.Lock()

        self.grabbed_image = None
        self.selection_rect = None

        # flags
        self.hold_left = False
        self.checked_border = False
        self.is_grabbing_window = False
        self.image_is_set = False
        self.anchor_showed = False
        self.neighbour_showed = False
        self.next_neighbour_requested = False
        self.solution_applied = False
        self.initial_image_updates = False
        self.zoom_processed = False

        # keyboard flags
        self.is_grabbing_window = False

        self.keyboard_input = 0
        self.ratio = 1
        self.last_eval_bucket = -1
        self.time_stamp = 0
        self.selected_pic = 0

        self.toolbar_color = 0
        self.toolbar_bg = 0

        self.image_offset = [0, 0]
        self.mouse_pos = [0, 0]

        self.pl_solution = {}

        self.showed_image_list = []
        self.final_solution = []
        self.current_image_list = []
        self.bank_image_list = []
        self.neighbour_ids = []
        self.file_names = []
        self.key_fragments = []
        self.key_list = []

        self.clicked = ""
        self.key_fragment_id = ""

        self.resize_event = None
        self.key_image = None
        self.select_anchor_running = None
        self.select_neighbour_running = None
        self.pl_solver_running = None
        self.select_anchor_done = None
        self.select_neighbour_done = None
        self.pl_solver_done = None

        self.the_layout = MDFloatLayout(md_bg_color=(0, 0, 0, 1))
        self.grid_layout = GridLayout()

        self.grid_layout.base_scale_factor = 1.0
        self.grid_layout.true_scale = 1.0
        self.grid_layout.scale_factor = self.grid_layout.base_scale_factor
        self.grid_layout.zoom_scale = Scale(x=self.grid_layout.scale_factor, y=self.grid_layout.scale_factor, origin=(0, 0))
        # PushMatrix()
        self.grid_layout.canvas.add(self.grid_layout.zoom_scale)

        # with self.grid_layout.canvas.after:
        #     PopMatrix()
        self.toolbar = MDTopAppBar()
        self.click_label = Label()
        self.main_layout = MainLayout()
        self.sidebar = MDCard(size_hint=(None, None), size=(64, Window.size[1] - 64), pos_hint={"right": 1, "down": 1},
                              md_bg_color=(0.1, 0.1, 0.1, 1))

        self.pause_play_button = ToggleButton(
            size_hint=(None, None),
            width=64, height=64,
            background_normal= os.path.join(path_dic['icons'], "play.png"),  # Initial icon
            background_down= os.path.join(path_dic['icons'], "pause.png")
        )

        self.anchor_button = Button(text="Select Anchor")
        self.show_button = Button(text="Next")
        self.neighbour_button = Button(text="Neighbour")
        self.pl_solver_button = Button(text="PL Solver")
        self.final_button = Button(text="Finish")
        self.progress_bar = ProgressBar()



        self.sidebar = MDCard(size_hint=(None, None), size=(64, Window.size[1] - 64), pos_hint={"right": 1, "down": 1},
                              md_bg_color=(0.1, 0.1, 0.1, 1))
        self.sidebar.col_grid = MDGridLayout()
        self.sidebar.col_grid.accept_button = ToggleButton()
        self.sidebar.col_grid.deny_button = ToggleButton()

        self.lock_apply_solution = False
        self.probability_matrix = None

        self.sandbox = None

        self.sandbox_group = None  # InstructionGroup holding the overlay
        self.sandbox_size = (1600, 800)  # default box size in pixels (was “cm” in your note)
        self.sandbox_center = None  # remembers last center used

        self.cm_to_px = 1.0

        # self.test_monkey = Widget3D('3D/untitled.obj', '3D/simple.glsl')

    def build(self):
        for m in get_monitors():
            if m == get_monitors()[0]:
                Window.left = m.x
                Window.top = m.y
                Window.size = (m.width, m.height)
        # the_layout = Scatter()
        self.main_layout.cols = 1

        Window.bind(
            on_motion=self.on_touch_move,
            on_touch_down=self.on_touch_down,
            on_key_down=self._on_keyboard_down,
            on_key_up=self.on_keyboard_up,
            on_resize=self.on_resize,
            on_touch_up = self.on_touch_up
        )

        self.main_layout.rows = 3
        self.main_layout.padding = [0, 0, 0, 0]
        self.main_layout.spacing = [0, 0]

        self.toolbar = MDTopAppBar(title="GUI", elevation=4)
        self.toolbar.size_hint_y = None
        self.toolbar.height = dp(64)  # pick your height
        self.toolbar.pos_hint = {"top": 1}  # stick to top of parent

        self.toolbar.orientation = "horizontal"

        self.anchor_button.bind(on_press=start_select_anchor)

        self.show_button.bind(on_press=get_next_neighbour)

        self.neighbour_button.bind(on_press=start_select_neighbour)

        self.final_button.bind(on_press=to_the_robot)
        # the_layout.add_widget(snackbar)

        self.pl_solver_button.bind(on_press=start_pl_solver)

        self.pause_play_button.bind(on_press=toggle_program_lock)

        # self.toolbar.left_action_items.append(["menu", lambda x: self.the_app.callback()])
        void_image = Image(source=os.path.join(path_dic['icons'], "Void Image.png"))
        self.toolbar.add_widget(self.progress_bar)
        self.toolbar.add_widget(void_image)
        self.progress_bar.max = 100
        self.progress_bar.value = 0

        self.toolbar.add_widget(self.pause_play_button)
        self.toolbar.add_widget(self.pl_solver_button)
        self.toolbar.add_widget(self.neighbour_button)
        self.toolbar.add_widget(self.anchor_button)
        self.toolbar.add_widget(self.show_button)

        self.anchor_button.size_hint_x = Window.size[0] / 8
        self.show_button.size_hint_x = Window.size[0] / 8
        self.neighbour_button.size_hint_x = Window.size[0] / 8
        self.pl_solver_button.size_hint_x = Window.size[0] / 8
        void_image.size_hint_x = Window.size[0] / 128
        self.progress_bar.size_hint_x = Window.size[0] / 2
        self.final_button.size_hint_x = Window.size[0] / 2

        self.main_layout.minimum_height = 1

        self.click_label.text = "Click on the pictures"
        self.click_label.size_hint_y = 0.1
        self.click_label.height = 0.1

        self.grid_layout.cols = 5
        self.grid_layout.size_hint = (1, 1)
        self.grid_layout.padding = Window.size[1]/16
        self.grid_layout.spacing = 0

        self.main_layout.add_widget(self.grid_layout)
        self.main_layout.add_widget(self.click_label)

        self.the_layout.add_widget(self.main_layout)

        # toggle_button = MDRaisedButton(text="Toggle Sidebar", size_hint=(None, None), size=(200, 50))
        # toggle_button.bind(on_release=self.toggle_sidebar)

        # Sidebar layout
        self.sidebar.image_name = Label(text="", size_hint=(None, None), size=(64, 32))

        self.sidebar.col_grid.cols = 2
        self.sidebar.col_grid.rows = 1
        self.sidebar.col_grid.label1 = Label(text="anchor", size_hint=(None, None), size=(44, 32))

        self.sidebar.col_grid.accept_button.text = "Accept"
        self.sidebar.col_grid.accept_button.size_hint_max_y = 50
        self.sidebar.col_grid.accept_button.background_color = (0, 1, 0, 1) #Green
        self.sidebar.col_grid.accept_button.disable = True

        self.sidebar.col_grid.accept_button.bind(on_press=self.image_accepted)

        self.sidebar.col_grid.deny_button.text = "Deny"
        self.sidebar.col_grid.deny_button.size_hint_max_y = 50
        self.sidebar.col_grid.deny_button.background_color = (0.8, 0.1, 0.1, 1) #Light Red
        self.sidebar.col_grid.deny_button.disable = True

        self.sidebar.col_grid.deny_button.bind(on_press=self.image_denied)

        self.sidebar.col_grid.label3 = Label(text="deny", size_hint=(None, None), size=(44, 32))
        self.sidebar.col_grid.add_widget(self.sidebar.col_grid.label1)

        self.sidebar.image_checkbox = CheckBox(size_hint=(None, None), size=(20, 32))
        self.sidebar.image_checkbox.color = (0.6,0.6,0.6,1)
        self.sidebar.image_checkbox.group = "image_properties"

        self.sidebar.image_checkbox.bind(active=self.on_checkbox_active)
        self.sidebar.col_grid.add_widget(self.sidebar.image_checkbox)

        self.sidebar.main_grid = MDGridLayout()

        self.sidebar.main_grid.cols = 1
        self.sidebar.main_grid.rows = 5

        self.add_anchor_option()

        self.sidebar.add_widget(self.sidebar.main_grid)
        self.sidebar.image_checkbox.state = "normal"  # False
        self.sidebar.image_checkbox.state = "down"  # True
        # self.sidebar.add_widget(MDRaisedButton(text="Option 1"))
        # self.sidebar.add_widget(MDRaisedButton(text="Option 2"))
        # self.sidebar.add_widget(MDRaisedButton(text="Option 3"))
        self.sidebar.opacity = 0  # Initially hidden

        # the_layout.add_widget(toggle_button)
        self.the_layout.add_widget(self.sidebar)
        self.the_layout.add_widget(self.toolbar, index = 0)

        self.anchor_button.disabled = False
        self.show_button.disabled = True
        self.neighbour_button.disabled = True
        self.pl_solver_button.disabled = True

        Clock.schedule_interval(self.checking_clock, graphic_freq)  # Graphic Internal Thread to communicate
        return self.the_layout

    def add_accept_deny_button(self):
        #clear sidebar
        self.sidebar.main_grid.clear_widgets()

        self.sidebar.main_grid.add_widget(self.sidebar.image_name)
        self.sidebar.main_grid.add_widget(self.sidebar.col_grid.accept_button)
        self.sidebar.main_grid.add_widget(self.sidebar.col_grid.deny_button)

    def add_anchor_option(self):
        #clear sidebar
        self.sidebar.main_grid.clear_widgets()

        self.sidebar.main_grid.add_widget(self.sidebar.image_name)
        self.sidebar.main_grid.add_widget(self.sidebar.col_grid)
        self.sidebar.main_grid.add_widget(self.sidebar.col_grid.accept_button)
        self.sidebar.main_grid.add_widget(self.sidebar.col_grid.deny_button)

    def image_accepted(self, instance):
        current_image = self.current_image_list[self.selected_pic - 1]
        if current_image.status == Status.NEUTRAL:
            self.set_image_accepted(current_image)
        elif current_image.status == Status.ACCEPTED:
            self.set_image_neutral(current_image)

    def set_image_accepted(self, current_image):
        current_image.status = Status.ACCEPTED
        neighbour = []
        if update_started:
            # couple = (current_image.name, current_image.position_memory)
            # back_end.set_p_elements(couple, self.image_offset)
            # current_image.set_anchor(True)
            neighbour = check_neighbouring_collision(current_image)
            # print('neighbour', neighbour)
            update_compatibility_matrix(current_image, neighbour, True)

        # for image in neighbour:
        #     print(image.name)

        self.set_sidebar_accepted()

    def set_sidebar_accepted(self):
        self.sidebar.col_grid.accept_button.state = 'down'
        self.sidebar.col_grid.deny_button.state = 'normal'
        self.sidebar.col_grid.accept_button.disabled = False
        self.sidebar.col_grid.deny_button.disabled = True

    def set_image_denied(self, current_image):
        # print("Denied")
        current_image.status = Status.DENIED
        neighbour = []
        if update_started:
            # couple = (current_image.name, current_image.position_memory)
            # back_end.set_p_elements(couple, self.image_offset)
            # current_image.set_anchor(False)
            neighbour = check_neighbouring_collision(current_image)
            # print('neighbour', neighbour)
            update_compatibility_matrix(current_image, neighbour, False)

        self.set_sidebar_denied()

    def set_sidebar_denied(self):
        self.sidebar.col_grid.accept_button.state = 'normal'
        self.sidebar.col_grid.deny_button.state = 'down'
        self.sidebar.col_grid.accept_button.disabled = True
        self.sidebar.col_grid.deny_button.disabled = False

    def set_image_neutral(self, current_image):
        # print("Neutral")
        current_image.status = Status.NEUTRAL

        self.set_sidebar_neutral()

    def set_sidebar_neutral(self):
        self.sidebar.col_grid.accept_button.state = 'normal'
        self.sidebar.col_grid.deny_button.state = 'normal'
        self.sidebar.col_grid.accept_button.disabled = False
        self.sidebar.col_grid.deny_button.disabled = False

    def image_denied(self, instance):
        current_image = self.current_image_list[self.selected_pic-1]
        if current_image.status == Status.NEUTRAL:
            self.set_image_denied(current_image)
        elif current_image.status == Status.DENIED:
            self.set_image_neutral(current_image)

    @mainthread
    def set_images(self, has_score, *args, **kwargs):
        scores = back_end.image_scores
        self.current_image_list = []
        # score_label.text = str(scores[i])
        self.file_names = back_end.image_names
        last_anchors = []
        if self.key_list is not []:
            for i, fragment in enumerate(reversed(self.key_list)):
                if fragment in self.file_names:
                    index = self.file_names.index(fragment)
                    scores.remove(scores[index])
                    self.file_names.remove(fragment)
                    back_end.image_numbers -= 1
            for i, fragment in enumerate(reversed(self.key_list)):
                scores.append(str(len(self.key_list) - i) + "-Anchor")
                self.file_names.append(fragment)
                back_end.image_numbers += 1
                # scores.append(0)

            # self.file_names.append(key_image.get_id())
            # back_end.image_numbers += 1
            # scores.append(0)
        for bank_image in self.bank_image_list:
            if bank_image.is_anchor:
                last_anchors.append(bank_image.name)
        for i in range(back_end.image_numbers):

            image = image_reader(i, scores[i], has_score)
            # image.fit_mode = "contain"
            if image.name in last_anchors:
                image.set_anchor(True)
            self.current_image_list.append(image)
        self.initial_image_updates = False

    def on_checkbox_active(self, checkbox, value):
        # print("here")
        if hasattr(self, 'grabbed_image') & (self.sidebar.opacity == 1):
            if self.grabbed_image is not None:
                self.grabbed_image.set_anchor(value)
                if value:
                    if update_started:
                        couple = (self.grabbed_image.name, self.grabbed_image.position_memory)
                        back_end.set_p_elements(couple, self.image_offset)
                        self.grabbed_image.set_anchor(True)
                else:
                    if update_started:
                        couple = (self.grabbed_image.name, self.grabbed_image.position_memory)
                        back_end.set_p_elements(couple, self.image_offset, False)
                        self.grabbed_image.set_anchor(False)

    def toggle_sidebar(self, on_off, true_false):
        # Toggle sidebar visibility
        if on_off:
            self.sidebar.opacity = 1

            if self.grabbed_image.is_selected:
                self.grabbed_image.lock_movement = True

            if self.grabbed_image.status == Status.NEUTRAL:
                self.set_sidebar_neutral()
            elif self.grabbed_image.status == Status.ACCEPTED:
                self.set_sidebar_accepted()
            elif self.grabbed_image.status == Status.DENIED:
                self.set_sidebar_denied()
            # self.sidebar.add_widget(self.test_monkey)
        else:
            self.sidebar.opacity = 0
            # self.sidebar.remove_widget(self.test_monkey)
        if true_false:
            self.sidebar.image_checkbox.state = "down"
        else:
            self.sidebar.image_checkbox.state = "normal"

    @mainthread
    def on_touch_down(self, window, touch, *args, **kwargs):
        if touch.button == 'left':
            self.start_pos = touch.pos
            self.hold_left = True

    @mainthread
    def on_touch_up(self, window, touch, *args, **kwargs):
        if touch.button == 'left':
            self.hold_left = False
            # if self.keyboard_input != 305:
            if hasattr(self, 'grabbed_image') and self.grabbed_image is not None:
                # if update_started:
                #     couple = (self.grabbed_image.name, self.grabbed_image.position_memory)
                #     back_end.set_p_elements(couple, self.image_offset)
                #     self.grabbed_image.is_anchor = True
                self.grabbed_image.set_is_grabbed(False)
                self.grabbed_image.deselect()
            if hasattr(self, 'selection_rect'):
                if self.selection_rect in self.grid_layout.canvas.children:
                    for image in reversed(self.current_image_list):
                        if check_collision(image, self.selection_rect):
                            image.select()
                    self.grid_layout.canvas.remove(self.selection_rect)
            if self.keyboard_input == 305:  # left ctrl
                for image in reversed(self.current_image_list):
                    if check_image_select(image, self.mouse_pos):
                        image.select_toggle()
                        break

    @mainthread
    def on_touch_move(self, window, pos, touch, *args, **kwargs):  # Mouse Listener
        window_size = Window.size
        self.mouse_pos = (window_size[0] * touch.spos[0], window_size[1] * touch.spos[1])
        if self.keyboard_input == 304:  # left shift
            self.is_grabbing_window = True
        if self.keyboard_input == 32: # space
            if touch.button == 'scrollup':  # scroll up is scrolling down :|
                if self.grabbed_image is not None:
                    self.grabbed_image.rotate(-1 * rotation_interval)  # its  2x God knows why
            elif touch.button == 'scrolldown':  # scrolldown is scrolling up :|
                if self.grabbed_image is not None:
                    self.grabbed_image.rotate(+1 * rotation_interval)  # its  2x God knows why

        # zooming functionality
        if self.keyboard_input == 308:  # left alt
            if not hasattr(self, 'zoom_processed') or not self.zoom_processed:
                if touch.button == 'scrollup' or touch.button == 'scrolldown':
                    if touch.button == 'scrollup':  # scroll up is scrolling down :|
                        # self.zoom_at_point(0.95, self.mouse_pos)
                        for image in self.current_image_list:
                            image.zoom_at_point(0.95, self.mouse_pos)
                        if self.sandbox is not None:
                            self.sandbox.zoom_at_point(0.95, self.mouse_pos)
                    elif touch.button == 'scrolldown':  # scrolldown is scrolling up :|
                        # self.zoom_at_point(1.05, self.mouse_pos)
                        for image in self.current_image_list:
                            image.zoom_at_point(1.05, self.mouse_pos)
                        if self.sandbox is not None:
                            self.sandbox.zoom_at_point(1.05, self.mouse_pos)
                    self.zoom_processed = True
            else:
                self.zoom_processed = False
        # end of zooming

        if touch.button == 'right':
            # if hasattr(self, 'grabbed_image') and self.grabbed_image is not None:
            #     self.grabbed_image.deselect()
            # self.grabbed_image = None
            self.toggle_sidebar(False, False)
            for image in self.current_image_list:
                if image.is_selected:
                    image.deselect()

        if touch.button == 'left':
            if touch.is_double_tap:
                if not hasattr(touch, 'double_tapped') or not touch.double_tapped:
                    touch.double_tapped = True
            if not self.hold_left:
                self.checked_border = False
                if self.keyboard_input != 305: # left ctrl
                    for image in reversed(self.current_image_list):
                        if not hasattr(touch, 'dragging') or not touch.dragging:
                            if check_image_select(image, self.mouse_pos):
                                check_collision_with_sandbox(image, self.sandbox_group)
                                # if hasattr(self, 'grabbed_image') and self.grabbed_image is not None:
                                #     self.grabbed_image.deselect()
                                self.grabbed_image = image
                                self.grabbed_image.set_is_grabbed(True)
                                self.grabbed_image.select()
                                self.grabbed_image.update_translate()

                                self.checked_border = True  # /todo
                                break
                    if not self.checked_border:
                        # if hasattr(self, 'grabbed_image') and self.grabbed_image is not None:
                        #     self.grabbed_image.deselect()
                        self.grabbed_image = None

            if self.hold_left:  # hold left
                if self.keyboard_input == 103:  # G
                    with self.grid_layout.canvas:
                        if not hasattr(self, 'selection_rect') or not (
                                self.selection_rect in self.grid_layout.canvas.children):
                            Color(0, 1, 0, 0.3)
                            rectangle_size_x = 10
                            rectangle_size_y = 10
                            self.selection_rect = Rectangle(pos=(touch.x, touch.y - rectangle_size_y),
                                                            size=(rectangle_size_x, rectangle_size_y))
                            self.selection_rect.pos = (touch.x, touch.y - rectangle_size_y)
                            self.selection_rect.size = (rectangle_size_x, rectangle_size_y)
                    if hasattr(self, 'selection_rect'):
                        self.selection_rect.size = (touch.x - self.start_pos[0], touch.y - self.start_pos[1])

                else:
                    if self.is_grabbing_window:  # left shift
                        if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
                            # Store the initial touch position
                            touch.offset_x = self.mouse_pos[0]
                            touch.offset_y = self.mouse_pos[1]

                        # Update the position of the layout based on the movement of the mouse
                        self.grid_layout.pos = (self.grid_layout.pos[0] + (self.mouse_pos[0] - touch.offset_x),
                                           self.grid_layout.pos[1] + (self.mouse_pos[1] - touch.offset_y))

                        # Update the touch position for the next move event
                        touch.offset_x = self.mouse_pos[0]
                        touch.offset_y = self.mouse_pos[1]
                        self.is_grabbing_window = False
                    else:
                        # for image in reversed(self.current_image_list):
                        #     if image.is_selected:
                        #         if not hasattr(touch, 'dragging') or not touch.dragging:
                        #             if check_image_select(image, mouse_pos):
                        #                 touch.dragging = True
                        #                 print('dragging')
                        #                 if hasattr(self, 'grabbed_image') and self.grabbed_image is not None:
                        #                     self.grabbed_image.deselect()
                        #                 self.grabbed_image = image
                        #                 self.grabbed_image.select()
                        #                 self.grabbed_image.update_translate()
                        #
                        #                 self.checked_border = True  # /todo
                        #                 break
                        if self.grabbed_image is not None and self.checked_border:
                            if not hasattr(touch, 'offsets'):
                                touch.offsets = {}
                                for image in self.current_image_list:
                                    if image.is_selected:
                                        touch.offsets[image.name] = [
                                            (self.mouse_pos[0]/image.zoom_scale.x - image.get_real_pos()[0]),
                                            (self.mouse_pos[1]/image.zoom_scale.x - image.get_real_pos()[1]),
                                        ]
                            for image in self.current_image_list:
                                if image.is_selected:
                                    offset = touch.offsets[image.name]
                                    image.translate((self.mouse_pos[0]/image.zoom_scale.x - offset[0]), (self.mouse_pos[1]/image.zoom_scale.x - offset[1]))
                        # if self.grabbed_image is not None and self.checked_border:
                        #     if not hasattr(touch, 'offset_x') or not hasattr(touch, 'offset_y'):
                        #         touch.offset_x = self.mouse_pos[0] - self.grabbed_image.get_real_pos()[0]
                        #         touch.offset_y = self.mouse_pos[1] - self.grabbed_image.get_real_pos()[1]
                        #     self.grabbed_image.translate(self.mouse_pos[0] - touch.offset_x, self.mouse_pos[1] - touch.offset_y)
                        #
                            if (not self.select_anchor_running) and (not self.select_neighbour_running) and (not self.select_neighbour_done):
                                self.key_fragment_id = str(self.grabbed_image.get_id())
                                self.key_image = self.grabbed_image
                            self.click_label.text = str(self.grabbed_image.name)
                            self.clicked = str(self.grabbed_image.image_number + 1)
                            if hasattr(touch, 'double_tapped'):
                                # temp_text = self.grabbed_image.name
                                # numbers = re.findall(r'\d+', temp_text)
                                # self.sidebar.image_name.text = '_'.join(numbers)
                                # self.toggle_sidebar(True, self.grabbed_image.is_anchor)
                                print(self.sidebar.image_name.text)
                                if self.grabbed_image.is_anchor:
                                    self.double_tap_anchor(self.grabbed_image, False)
                                else:
                                    self.double_tap_anchor(self.grabbed_image, True)

    def double_tap_anchor(self, image, value):
        if image is not None:
            image.set_anchor(value)
            if value:
                if update_started:
                    couple = (image.name, image.position_memory)
                    back_end.set_p_elements(couple, self.image_offset)
                    image.set_anchor(True)
            else:
                if update_started:
                    couple = (image.name, image.position_memory)
                    back_end.set_p_elements(couple, self.image_offset, False)
                    image.set_anchor(False)

    @mainthread
    def on_keyboard_up(self, instance, keyboard, keycode):  # Keyboard up Listener
        self.keyboard_input = 0
        if keyboard is not None:
            if keyboard == 32:
                self.keyboard_input = 0
            if keyboard == 304:
                self.keyboard_input = 0
            if keyboard == 308:
                self.keyboard_input = 0

    @mainthread
    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):  # Keyboard down Listener
        self.keyboard_input = keyboard
        # zoom reset
        if self.keyboard_input == 122:  # z
            self.zoom_reset()

    def zoom_reset(self):
        for image in self.current_image_list:
            image.zoom_reset()
        if self.sandbox is not None:
            self.sandbox.zoom_reset()

    @mainthread
    def on_resize(self, *args):
        if self.sandbox is not None:
            self.bounding_box(is_set=False)
            self.bounding_box(is_set=True)
        self.sidebar.size = (64, Window.size[1] - self.toolbar.size[1])
        if (self.select_anchor_done is not None) & (self.select_neighbour_done is not None) & (self.pl_solver_done is not None):
            if self.select_anchor_done & self.select_neighbour_done & self.pl_solver_done:
                if self.resize_event is None:
                    # self.apply_resize()
                    self.resize_event = Clock.schedule_interval(self.apply_resize_throttled, graphic_freq)

    def apply_resize_throttled(self, dt):
        self.apply_resize()  # Call the actual resize logic
        Clock.unschedule(self.resize_event)  # Unschedule the event after the update
        self.resize_event = None

    @mainthread
    def show_images(self, *args, **kwargs):
        if len(self.showed_image_list) != 0:
            self.clear_images()

        self.showed_image_list = []
        self.time_stamp = time.time()
        self.toggle_sidebar(False, False)
        for i in range(len(self.current_image_list)):
            self.showed_image_list.append(self.current_image_list[i].grid)
            self.grid_layout.add_widget(self.showed_image_list[i])
            self.time_stamp = time.time()

    @mainthread
    def bounding_box(self, is_set, size=(100, 100), center=(0, 0), padding=0, outline_width=2):
        """
        Draw (or remove) a transparent-centered sandbox box over the UI.
        - is_set=True  -> draw/refresh the overlay
          is_set=False -> remove the overlay
        - size:   (w, h) of the box in pixels; defaults to self.sandbox_size
        - center: (cx, cy) in window coords; defaults to window center
        - padding: extra padding around the box (pixels)
        """
        # center = [Window.size[0] / 2, Window.size[1] / 2]
        # if self.sandbox is None and is_set:
        #     self.sandbox_size = size
        #     self.sandbox = SandBox(size, center, Window.size, self.cm_to_px)
        #     self.grid_layout.canvas.after.add(self.sandbox)
        # elif not is_set and self.sandbox is not None:
        #         self.grid_layout.canvas.after.remove(self.sandbox)
        #         self.sandbox = None
        return

    def clear_images(self, *args, **kwargs):
        for i in range(len(self.showed_image_list)):
            self.grid_layout.remove_widget(self.showed_image_list[i])
        self.showed_image_list = []

    @mainthread
    def add_final_button(self):
        self.toolbar.remove_widget(self.anchor_button)
        self.toolbar.add_widget(self.final_button)
        self.toolbar.remove_widget(self.pl_solver_button)
        self.toolbar.remove_widget(self.neighbour_button)
        self.toolbar.remove_widget(self.show_button)

    @mainthread
    def apply_resize(self):
        center = [Window.size[0] / 2, Window.size[1] / 2]
        # center = [0, 0]
        bank_offset = self.image_offset
        for image in self.current_image_list:
            positions = np.array(image.position_memory)

            positions = [positions[0] - bank_offset[0],
                         positions[1] - bank_offset[1],
                         positions[2]]
            #
            self.image_offset = image.update_offset(center)
            #
            positions = [positions[0] + self.image_offset[0],
                         positions[1] + self.image_offset[1],
                         positions[2]]

            # positions = current_positions[image_id]
            position = np.array([positions[0], positions[1]])

            image.update_positions(position, positions[2])
            # image.zoom_reset()

    @mainthread
    def apply_solution(self, is_final = True):
        # Calculate the center of the window
        all_anchored = True
        center = [Window.size[0] / 2, Window.size[1] / 2]

        for image in self.current_image_list:
            image.remove_score()
            image_id = image.get_id()

            if not image.is_anchor:
                all_anchored = False

            if image_id in self.pl_solution:
                if is_final:
                    if self.probability_matrix is not None:
                        if self.probability_matrix[image_id] > 0.99:
                            image.set_anchor(True)
                positions = self.pl_solution[image_id]
                position = np.array([positions[1], (-1 * positions[0])])  # fix the coordinate system

                image.update_ratio()

                # self.image_offset_x = center[0] - image.parent.size[0] / 2 - (image.parent.pos[0] - image.pos[0]) / 2
                # self.image_offset_y = center[1] - image.parent.size[1] / 2 - (image.parent.pos[1] - image.pos[1]) / 2

                # centering the anchor and moving others, image.parent (it's canvas) is responsible for positioning
                self.image_offset = image.update_offset(center)

                # ratio will apply in update_positions function
                new_positions = np.array(
                    [position[0] + self.image_offset[0], position[1] + self.image_offset[1]])
                if (not image.is_grabbed) and (not self.lock_apply_solution):
                    if self.probability_matrix is not None:
                        set_probabilities(self.pl_solution, self.probability_matrix, image)
                    image.update_positions(new_positions, positions[2])
            else:
                image.update_positions([-1500, -1500], 0)
        if all_anchored:
            app.bounding_box(is_set=False)
            back_end.kill_puzzle_solver()

    def checking_clock(self, *args, **kwargs):
        self.communicate_thread_lock.acquire()
        self.toolbar_changes(self.toolbar_color)

        clicked = self.clicked

        if (clicked.isdigit()) & (self.selected_pic == 0):
            self.neighbour_button.disabled = False
            self.selected_pic = int(clicked)
        if clicked.isdigit():
            self.selected_pic = int(clicked)
        if back_end.pl_solver_running:
            universal_zoom(1)
        # if not self.anchor_showed:
        #     if (back_end.get_select_anchor_done()) & (len(self.current_image_list) == 0):
        #         self.show_anchors(self)
        # if ((back_end.get_select_anchor_done()) & (len(self.current_image_list) == 0) &
        #       (back_end.get_select_neighbour_done()) & (not self.neighbour_showed)):
        #     self.show_neighbours()
        if ((back_end.get_select_anchor_done()) & (len(self.current_image_list) == 0) &
              (back_end.get_select_neighbour_done()) & self.neighbour_showed & self.next_neighbour_requested):
            self.set_images(1)
            self.show_images(self)
            self.next_neighbour_requested = False
        # elif ((back_end.get_select_anchor_done()) & (back_end.get_select_neighbour_done()) & self.neighbour_showed &
        #       back_end.get_pl_solver_done() & (not self.solution_applied)):
        #     self.show_solutions()
        self.communicate_thread_lock.release()

    def show_solutions(self):
        self.sidebar.col_grid.label1.text = "Accept"
        iteration = back_end.get_iteration()
        self.pl_solution, original_answer, probability = back_end.get_pl_solution()
        back_end.calculate_results(original_answer, None, iteration, -1)
        answer = self.pl_solution.copy()
        save_demo_parameters(answer, probability, iteration = "final", is_end=True)
        self.apply_solution(True)
        self.solution_applied = True
        # self.pl_solver_button.disabled = True
        # self.show_button.text = 'Next Loop'
        # self.show_button.disabled = False
        # self.neighbour_button.disabled = True
        self.add_final_button()


    def show_neighbours(self):
        self.set_images(1)
        self.show_images(self)
        self.neighbour_showed = True
        self.neighbour_button.disabled = True
        self.show_button.disabled = False
        self.pl_solver_button.disabled = False

    def show_anchors(self):
        self.set_images(1)
        self.show_images(self)
        self.anchor_showed = True
        self.anchor_button.disabled = True

    def toolbar_changes(self, color):
        if color == 0:
            if self.toolbar_bg != 0:
                self.toolbar.md_bg_color = (0.678431373, 0.847058824, 0.901960784, 1)  # Set Toolbar Blue
                self.toolbar_bg = 0
        elif color == -1:
            if self.toolbar_bg != -1:
                self.toolbar.md_bg_color = (0.545098039, 0, 0, 1)  # Set Toolbar Red
                self.toolbar_bg = -1
        elif color == 1:
            if self.toolbar_bg != 1:
                self.toolbar.md_bg_color = (0.141176471, 0.529411765, 0.129411765, 1)  # Set Toolbar Green
                self.toolbar_bg = 1

    def callback(self):
        # start_select_anchor(self)
        return

    def zoom_at_point(self, factor, origin):
        new_scale = self.grid_layout.scale_factor * factor

        last_origin = self.grid_layout.zoom_scale.origin
        origin_diff = (origin[0] - last_origin[0], origin[1] - last_origin[1])

        if (self.grid_layout.scale_factor < self.grid_layout.true_scale < new_scale) or (self.grid_layout.scale_factor > self.grid_layout.true_scale > new_scale):
            new_scale = self.grid_layout.true_scale

        if (new_scale > 0.95) & (new_scale < 1.05):
            new_scale = 1.0
        if new_scale < 0.5:
            new_scale = 0.5
        origin_diff = (origin_diff[0] / new_scale, origin_diff[1] / new_scale)
        # Update the origin of the scale to the local mouse position
        self.grid_layout.zoom_scale.origin = (last_origin[0] + origin_diff[0], last_origin[1] + origin_diff[1])

        # Adjust the scale factor
        self.grid_layout.scale_factor = new_scale
        self.grid_layout.zoom_scale.x = self.grid_layout.scale_factor
        self.grid_layout.zoom_scale.y = self.grid_layout.scale_factor
        self.grid_layout.base_scale_factor = self.grid_layout.scale_factor

    @mainthread
    def final_button_running(self):
        self.final_button.text = "Generating..."
        # make background color visible in all states
        self.final_button.background_normal = ''
        self.final_button.background_down = ''
        self.final_button.background_disabled_normal = ''
        self.final_button.background_disabled_down = ''

        # text colors
        self.final_button.color = (1, 1, 1, 1)  # normal text
        self.final_button.disabled_color = (1, 1, 1, 1)  # text while disabled

        # background color while running
        self.final_button.background_color = (1, 0, 0, 1)  # red
        self.final_button.disabled = True

    @mainthread
    def final_button_finished(self):
        self.final_button.text = "Finish"
        self.final_button.disabled = False
        dummy = Button()
        self.final_button.background_normal = dummy.background_normal
        self.final_button.background_down = dummy.background_down
        self.final_button.background_disabled_normal = dummy.background_disabled_normal
        self.final_button.background_disabled_down = dummy.background_disabled_down
        self.final_button.background_color = dummy.background_color
        self.final_button.color = dummy.color
        self.final_button.disabled_color = dummy.disabled_color

def start_select_anchor(self):
    global back_end
    app.image_is_set = False
    back_end.start_anchor_thread()

def to_the_robot(self):
    app.final_button_running()
    # Start long task in background
    threading.Thread(target=generate_placement).start()

def generate_placement():
    solved_pieces = get_screen_pieces(only_solved=False)
    print("solved_pieces", solved_pieces)
    app.pl_solution, original_answer, probability = back_end.get_pl_solution()
    # answer = loop_finalization(solved_pieces)
    answer_dic = {}
    for i in range(len(solved_pieces)):
        piece_id = solved_pieces[i][0]
        position = solved_pieces[i][1]
        position = [position[0], position[1], position[2]]
        answer_dic[piece_id] = position
    answer_dic = back_end.scale_solution(answer_dic)
    for piece_id, position in answer_dic.items():
        old_position = answer_dic[piece_id].copy()
        position[0] = -1 * old_position[1]
        position[1] = old_position[0]


    print("answer", answer_dic)

    save_demo_parameters(answer_dic, probability, iteration="final", is_end=True)
    for piece_id, position in answer_dic.items():
        old_position = answer_dic[piece_id].copy()
        position[0] = old_position[1]
        position[1] = -1 * old_position[0]
    # for piece_id, position in answer_dic.items():
    #     old_position = answer_dic[piece_id].copy()
    #     position[0] = old_position[1]
    #     position[1] = old_position[0]

    back_end.generate_placement_file(answer_dic)
    app.final_button_finished()

def start_select_neighbour(self):
    app.key_list = []
    app.key_fragments = []

    for image in app.current_image_list:
        if image.is_anchor:
            app.key_fragments.append(image)
            app.key_list.append(image.get_id())
        elif image.get_id() == app.key_fragment_id:
            app.key_fragments.append(image)
            app.key_list.append(image.get_id())
            image.set_anchor(True)
            # app.key_list.append(image.get_id())

    # # app.key_fragments = [app.key_fragment_id]
    # for piece in app.final_solution:
    #     if piece[0] != app.key_fragment_id:
    #         app.key_list.append(piece[0])
    #         # app.key_list.append(piece[0])

    if len(app.key_fragments) > 1:
        target_position = [0, 0]
        initial_position = app.key_fragments[0].position_memory
        initial_position = (initial_position[0],
                            initial_position[1],
                            initial_position[2])

        # offset = (target_position[0] - initial_position[0],
        #           target_position[1] - initial_position[1],
        #           0)

        bank_array = []
        for solution in app.final_solution:
            bank_array.append(solution[0])

        xy_step, theta_step = back_end.extract_steps()

        for key_fragment in app.key_fragments:
            if key_fragment.name in bank_array:
                break
            # key_fragment.update_ratio()
            image_ratio = key_fragment.ratio

            offset = key_fragment.update_offset(center=[Window.size[0] / 2, Window.size[1] / 2])

            pos = [key_fragment.position_memory[0],
                   key_fragment.position_memory[1],
                   key_fragment.position_memory[2]]

            # new_position = (pos[0] + offset[0],
            #                 pos[1] + offset[1],
            #                 pos[2])
            # new_position = [new_position[0]/image_ratio[0],
            #                 new_position[1]/image_ratio[1],
            #                 new_position[2]]

            pos = back_end.reverse_offset(pos, offset)
            pos = back_end.scale_to_solver(xy_step, theta_step, pos, path_dic)

            pos = [pos[0], pos[1], pos[2]]
            app.final_solution.append([key_fragment.get_id(), pos])
            pass
    back_end.key_fragment = app.key_fragment_id
    app.image_is_set = False
    app.bank_image_list = app.current_image_list
    app.current_image_list = []
    back_end.start_neighbour_thread(app.key_list)
    app.show_button.disabled = True
    app.neighbour_button.disabled = True

def toggle_program_lock(self):
    if self.state == 'down':
        # self.background_normal = 'Icons/play.png'  # Change to play icon
        app.lock_apply_solution = True
        back_end.solver_toggle_lock(True)
    else:
        # self.background_normal = 'Icons/pause.png'  # Change to pause icon
        app.lock_apply_solution = False
        back_end.solver_toggle_lock(False)

def start_pl_solver(self):
    global universal_zoom_applied
    global update_started
    app.show_button.disabled = True
    app.pl_solver_button.disabled = True

    anchor_pos = app.key_image.position_memory
    app.bounding_box(True, size=(600, 600))

    for i in range(0, len(app.current_image_list)):
        if not (app.current_image_list[i].get_id() == app.key_image.get_id()):
            app.neighbour_ids.append(app.current_image_list[i].get_id())
    back_end.neighbour_ids = app.neighbour_ids
    # self.image_is_set = False
    # app.current_image_list = []
    update_started = False
    solved_piece = back_end.solved_pieces + app.final_solution
    # print("solved_piece", solved_piece)
    app.last_eval_bucket = -1
    back_end.start_pl_solver_thread(back_end.key_fragment, back_end.neighbour_ids, solved_piece)
    universal_zoom_applied = False

def get_next_neighbour(self, *args, **kwargs):
    if not app.solution_applied:
        boolean, next_neighbours = back_end.get_next_neighbour(app.click_label.text)
        if boolean:
            app.image_is_set = False
            app.current_image_list = []
            app.next_neighbour_requested = True
    else:
        solved_pieces = get_screen_pieces()

        # build_meta_fragment(solved_pieces)

        loop_finalization(solved_pieces)

        app.show_button.disabled = True
        app.neighbour_button.disabled = False
        app.neighbour_showed = False
        back_end.set_pl_solver_done(False)
        app.solution_applied = False

def loop_finalization(solved_pieces):
    app.final_solution = back_end.loop_finalization(solved_pieces, app.image_offset)
    return app.final_solution
    # neighbour_test = ['piece_0006.png']
    # back_end.puzzle_solver_test_function(app.final_solution, neighbour_test)

def check_neighbouring_collision(grabbed_image):
    neighbors = []
    for image in app.current_image_list:
        # Skip checking the same piece
        if image.name == grabbed_image.name:
            continue
        else:
            if back_end.are_neighbors(grabbed_image, image):
                neighbors.append(image)
    return neighbors

def update_compatibility_matrix(current_image, neighbors, value):
    offset = current_image.update_offset(center=[Window.size[0] / 2, Window.size[1] / 2])
    if not value:
        back_end.removed_from_locked(current_image)
    for image in neighbors:
        if image.is_anchor:
            current_image_pos = current_image.position_memory
            image_pos = image.position_memory

            # Convert the masks to binary images
            current_image_mask = current_image.mask
            image_mask = image.mask

            # Find the contours of the masks
            contours_current, _ = cv2.findContours(current_image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_image, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            print("calculating")

            # Calculate the minimum distance between the contours
            min_distance = float('inf')
            for cnt1 in contours_current:
                for cnt2 in contours_image:
                    for point1 in cnt1:
                        for point2 in cnt2:
                            dist = np.linalg.norm(point1 - point2)
                            if dist < min_distance:
                                min_distance = dist

            print("Minimum distance between masks:", min_distance)

            back_end.set_cm_elements(current_image.name, image.name, current_image_pos, image_pos, offset, value)

def set_probabilities(answer, probability, image):
    image.remove_score()
    image_id = image.get_id()

    if image_id in answer:
        image.set_probability(probability[image_id])

update_counter = 0
update_freq = 1 # in seconds
update_started = False

def extract_final_number(filename):
    match = re.findall(r'(\d+)(?=\.png$)', filename)
    return int(match[-1]) if match else 0

def save_demo_parameters(answer, probability, iteration, is_end):
    final_list = []

    if not is_end:
        final_list.append(f"iteration: {iteration}")

    # Sort by final numeric suffix
    sorted_images = sorted(answer.keys(), key=extract_final_number)

    for image in sorted_images:
        image_without_png = image.replace('.png', '')
        position = answer[image]
        prob = probability[image][0] * 100
        line = f"{image_without_png} {position[0]} {position[1]} {position[2]} {prob}"
        final_list.append(line)

    if not is_end:
        file_path = os.path.join(path_dic['cache_path'], 'HIL_results_parameters.txt')
        with open(file_path, 'a') as f:
            np.savetxt(f, final_list, fmt='%s')
    else:
        np.savetxt(os.path.join(path_dic['cache_path'], 'final_result.txt'),
                   final_list, fmt='%s')

    return final_list

def save_parameters_to_json(answer, probability, process, filename="API-example.json"):
    # Convert numpy arrays to lists for JSON serialization
    cache = path_dic['cache_path']
    file_path = os.path.join(cache, filename)
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

        # Save to JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

def communicate_thread():  # communication thread, to communicate between UI, Graphic and BackEnd
    global update_counter
    global update_started
    answer = None
    probability = None
    while True:
        app.communicate_thread_lock.acquire()
        iteration = back_end.get_iteration()
        if answer is not None and probability is not None and iteration is not None and iteration > 4:
            bucket = iteration // 5
            if bucket > app.last_eval_bucket:
                back_end.calculate_results(answer, probability, iteration, bucket)
        if update_counter>=(1/communication_freq)*update_freq:
            answer, probability, process, iteration = back_end.get_solution_dict()
            # save_parameters_to_json(answer, probability, process)
            if answer is not None:
                save_demo_parameters(answer, probability, iteration, is_end=False)
                # back_end.save_results(answer)
                #     try:
                #         print(app.pl_solution[image])
                #     except KeyError:
                #         continue
                average_thresh_factor = 0.00
                app.pl_solution = answer
                if not app.lock_apply_solution:
                    answer = back_end.throw_away_1(answer, probability, average_thresh_factor)
                    app.pl_solution = answer
                app.probability_matrix = probability
                app.apply_solution(False)
                # save_parameters_to_json(answer, probability, process)
                update_started = True
                app.progress_bar.value = np.round(process * 100)
            update_counter = 0
        update_counter += 1
        app.select_anchor_running = back_end.get_select_anchor_running()
        app.select_neighbour_running = back_end.get_select_neighbour_running()
        app.pl_solver_running = back_end.get_pl_solver_running()
        app.select_neighbour_done = back_end.get_select_neighbour_done()
        app.select_anchor_done = back_end.get_select_anchor_done()
        app.pl_solver_done = back_end.get_pl_solver_done()

        if app.select_anchor_running is not None:
            if app.select_anchor_running:
                app.toolbar_color = -1
            else:
                app.toolbar_color = 1
        if app.select_anchor_done:
            app.toolbar_color = 1
        if app.select_neighbour_running is not None:
            if app.select_neighbour_running:
                app.toolbar_color = -1
            else:
                app.toolbar_color = 1
        else:
            app.toolbar_color = 0
        app.communicate_thread_lock.release()
        time.sleep(communication_freq)  # Thread sleep timerfasd

universal_zoom_applied = False
def universal_zoom(factor):
    global universal_zoom_applied
    if not universal_zoom_applied and update_started:
        # universal_center = calculate_universal_center()
        window_size = Window.size
        universal_center = [window_size[0]/2, window_size[1]/2]
        for image in app.current_image_list:
            image.zoom_default(factor,universal_center)
        universal_zoom_applied = True

def calculate_universal_center():
    universal_center = [0, 0]
    image_number = len(app.current_image_list)
    for image in app.current_image_list:
        universal_center[0] += image.center[0]
        universal_center[1] += image.center[1]
    universal_center[0] = universal_center[0]/image_number
    universal_center[1] = universal_center[1]/image_number
    return universal_center

def check_collision(image, rectangle):
    bounding_box = image.extract_bounding_box()
    top, left, bottom, right = bounding_box

    if rectangle.size[0] >= 0:
        rec_left = rectangle.pos[0]
        rec_right = rectangle.pos[0] + rectangle.size[0]
    else:
        rec_left = rectangle.pos[0] + rectangle.size[0]
        rec_right = rectangle.pos[0]

    if rectangle.size[1] >= 0:
        rec_bottom = rectangle.pos[1]
        rec_top = rectangle.pos[1] + rectangle.size[1]
    else:
        rec_bottom = rectangle.pos[1] + rectangle.size[1]
        rec_top = rectangle.pos[1]

    rec_right_top = [rec_right, rec_top]
    rec_left_bottom = [rec_left, rec_bottom]
    # zoom = [app.grid_layout.zoom_scale.x, app.grid_layout.zoom_scale.y, app.grid_layout.zoom_scale.origin]
    rec_right_top = image.map_mouse_pos_pixel(rec_right_top)
    rec_left_bottom = image.map_mouse_pos_pixel(rec_left_bottom)

    if not (top <= rec_right_top[1] and right <= rec_right_top[0] and bottom >= rec_left_bottom[1] and left >= rec_left_bottom[0]):
        return False
    return True

def check_collision_with_sandbox(image, sandbox_rectangle):
    if app.sandbox is not None:
        print("sandbox rectangle")
    else:
        print("No sandbox available.")

def check_image_select(image, mouse_pos):
    if image.collides(mouse_pos):
        if not app.checked_border:
            pixel = image.map_mouse_pos_pixel(mouse_pos)
            if pixel[0]<0 or pixel[1] < 0:
                return False
            return image.check_mask(pixel)
    return False

def image_reader(image_number, score, has_score, is_anchor=False):
    name = app.file_names[image_number]
    app.click_label.color = (1, 0, 1, 1)
    image_path = path_dic['image_path']
    mask_path = path_dic['mask_path']
    image = MovableImage(image_path, mask_path, app.click_label, score, image_number, has_score, name, 0, is_anchor)

    return image

def eroding(directory_path):
    eroded_dir_path = os.path.join(directory_path, "Eroded")

    # Create the 'Eroded' directory if it doesn't exist
    if not os.path.exists(eroded_dir_path):
        os.makedirs(eroded_dir_path)

    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory_path, file_name)
            image = cv2.imread(file_path)
            save_path = os.path.join(eroded_dir_path, file_name)  # Corrected save path
            file_name_without_extension = os.path.splitext(file_name)[0]
            # name of the file without extension
            eroded_image = run_erode(image, file_name_without_extension)
            cv2.imwrite(save_path, eroded_image)

def copy_to_cache(file_extension='.png'):
    source = path_dic['pieces_path']
    cache = path_dic['cache_path']
    # Remove the cache folder and its contents if it exists
    if os.path.exists(cache):
        shutil.rmtree(cache)

    # Recreate an empty cache folder
    os.makedirs(cache)
    if not os.path.exists(cache):
        os.makedirs(path_dic['cache_path'])
    for file_name in os.listdir(source):
        if file_name.endswith(file_extension):
            source_path = os.path.join(source, file_name)
            cache_path = os.path.join(cache, file_name)
            shutil.copy2(source_path, cache_path)
    # remove_image_from_cache('piece_0000')

def erode_data():
    file = open("GUI/DataBase/Archive/CM_color_border20.npy")
    f = "GUI/DataBase/Archive/CM_color_border20.npy"
    mmapped_array = np.load(f, mmap_mode='r')
    eroding_path = os.getcwd() + "/GUI/DataBase/Dafne/image_00000_1/pieces"
    eroding(eroding_path)

def transparent_data(path):
    for img in os.listdir(path):
        if img.endswith('.png'):
            input_path = os.path.join(path, img)
            img_t = transparent(input_path)
            output_image = img.split('.')[0] + ".png"
            output_path = os.path.join(path, output_image)
            cv2.imwrite(output_path, img_t)

def transparent(img):
    src = cv2.imread(img, 1)

    if src is None:
        print(f"Error: Unable to load image '{img}'. Please check the file path.")
        return None

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    _, alpha = cv2.threshold(tmp, 15, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(src)

    rgba = [b, g, r, alpha]

    dst = cv2.merge(rgba, 4)

    # kernel_size = 3
    # dst = cv2.medianBlur(dst, kernel_size)

    # dst = cv2.medianBlur(dst, kernel_size)

    return dst

def read_ground_truth():
    ground_truth = path_dic['ground_truth']
    # if ground_truth != "":
    #     with open(ground_truth, 'r') as file:
    #         lines = file.readlines()

def get_bounding_box(size, rotation):
    w, h = size
    # Convert rotation to radians
    rotation = math.radians(rotation)

    # Calculate bounding box using rotation matrix
    new_w = abs(w * math.cos(rotation)) + abs(h * math.sin(rotation))
    new_h = abs(w * math.sin(rotation)) + abs(h * math.cos(rotation))

    return int(new_w), int(new_h)

def calculate_canvas_size():
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    cache = path_dic['cache_path']

    for image in app.current_image_list:
        img_path = image.get_id()
        img_path = os.path.join(cache, img_path)
        position = image.position_memory
        x, y, rotation = int(position[0]), int(position[1]), int(position[2])
        y = -1 * y  # Invert y-axis

        # Open the puzzle piece to get its size
        piece = PILImage.open(img_path)
        original_size = piece.size

        # Calculate the bounding box after rotation
        rotated_size = get_bounding_box(original_size, rotation)

        # Calculate the corners of the bounding box based on the center position
        min_x = min(min_x, x - rotated_size[0] // 2)
        min_y = min(min_y, y - rotated_size[1] // 2)
        max_x = max(max_x, x + rotated_size[0] // 2)
        max_y = max(max_y, y + rotated_size[1] // 2)

    # Return the width and height of the bounding box
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    return int(canvas_width), int(canvas_height), int(min_x), int(min_y)

def get_screen_pieces(only_solved = True):
    solved_pieces = []
    for image in app.current_image_list:
        if image.is_anchor or not only_solved:
            image_id = image.get_id()
            position = image.position_memory
            x, y, rotation = int(position[0]), int(position[1]), int(position[2])
            y = y  # Invert y-axis
            solved_pieces.append((image_id, [x, y, rotation]))

    return solved_pieces

def build_meta_fragment(solved_pieces, canvas_size=(1000, 1000)):
    cache = path_dic['cache_path']

    canvas_width, canvas_height, offset_x, offset_y = calculate_canvas_size()
    canvas = PILImage.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))

    name = '_'

    for image in solved_pieces:
        image_id = image[0]
        img_path = image_id
        name = name + img_path[:-4] + "_"
        img_path = os.path.join(cache, img_path)
        x, y, rotation = image[1][0], image[1][1], image[1][2]
        y = -1 * y  # Invert y-axis

        # Open the puzzle piece
        piece = PILImage.open(img_path).convert('RGBA')

        # Rotate the piece around its center
        rotated_piece = piece.rotate(rotation, expand=True)

        # Calculate new position to paste based on the center
        center_x, center_y = rotated_piece.size[0] // 2, rotated_piece.size[1] // 2

        paste_position = ((x - center_x - offset_x), (y - center_y - offset_y))

        # Paste the rotated piece onto the canvas
        canvas.paste(rotated_piece, paste_position, rotated_piece)
        # remove_image_from_cache(image.get_id())
    name = name + ".png"
    output_path = cache + name
    canvas.save(output_path)

def remove_image_from_cache(image_id):
    cache = path_dic['cache_path']
    file_name = image_id
    file_path = os.path.join(cache, file_name)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)  # Remove the file
            print(f"Removed: {file_name} from {cache}")
        except Exception as e:
            print(f"Error removing {file_name}: {e}")
    else:
        print(f"File {file_name} does not exist in {cache}")

def get_setting():
    global path_dic
    global rotation_interval
    global back_end
    global backend_path

    path_dic, rotation_intervals, backend_path = back_end.setting("Setting.yaml")
    rotation_interval = float(rotation_intervals) / 2

    copy_to_cache()

if __name__ == '__main__':
    get_setting()

    Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
    # Config.set('graphics', 'fullscreen', '1')  # or '1'
    # Config.set('graphics', 'resizable', False)


    read_ground_truth()

    transparent_path = os.getcwd() + "/GUI/pieces/"
    transparent_data(transparent_path)

    # erode_data()

    app = GUIApp()
    back_end.set_main_app(app)

    test_thread = threading.Thread(target=communicate_thread, daemon=True)
    app.test_thread_started = True
    test_thread.start()

    app.run()
