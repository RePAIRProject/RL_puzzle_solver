import os

from kivy import Config
from kivy.clock import Clock, mainthread
from kivymd.app import MDApp
import numpy as np
import math
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar

from kivy.core.window import Window

import Back_End as back_end

from RL_puzzle_solver.puzzle_utils.puzzle_gen.generator import run_erode

import threading
import time

import shutil

import cv2

from MoveableImage import MovableImage

Window.clearcolor = (0, 0, 0, 0)

backend_path = os.getcwd() + "/GUI/DataBase/Images/RePAIR_plaque_2/"
image_path = ""
mask_path = ""
path_dic = None
rotation_interval = 0.5

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
next_neighbour_requested = False
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
    clicked = ""
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

        show_button = Button(text="Next")
        show_button.bind(on_press=self.get_next_neighbour)
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

        scores = back_end.image_scores
        current_image_list = []
        # score_label.text = str(scores[i])
        backend_path = backend_path
        file_names = back_end.image_names

        if key_image is not None:
            scores.append("Anchor")
            file_names.append(key_image.get_id())
            back_end.image_numbers += 1
            scores.append(0)
        for i in range(back_end.image_numbers):

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
        global rotation_interval

        scrolling = 0
        if keyboard_input == 304:
            self.is_grabbing_window = True
        if keyboard_input == 305:
            if touch.button == 'scrollup':  # scroll up is scrolling down :|
                if grabbed_image is not None:
                    grabbed_image.rotate(-1 * rotation_interval)  # its  2x God knows why
            elif touch.button == 'scrolldown':  # scrolldown is scrolling up :|
                if grabbed_image is not None:
                    grabbed_image.rotate(+1 * rotation_interval)  # its  2x God knows why

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
                                                            (image.texture_size[1]/image.norm_image_size[1]),
                                                            image.angle)

                                if image.check_mask(pixel):
                                    grabbed_image = image
                                    grabbed_image.update_translate()
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
                    # x_y = (mouse_pos[0] - touch.offset_x, mouse_pos[1] - touch.offset_y)
                    # r = np.array([1,1])
                    grabbed_image.translate(mouse_pos[0] - touch.offset_x, mouse_pos[1] - touch.offset_y)
                    # grabbed_image.update(x_y, r, 0)
                    if solution_applied:
                        grabbed_image

                    if (not select_anchor_running) and (not select_neighbour_running) and (not select_neighbour_done):
                        key_fragment_id = str(grabbed_image.get_id())
                        key_image = grabbed_image
                    # click_label.text = str(grabbed_image.get_number() + 1)
                    click_label.text = str(grabbed_image.get_name())
                    self.clicked = str(grabbed_image.get_number() + 1)
                else:
                    if self.is_grabbing_window:  # left shift
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
                self.apply_resize()

    @mainthread
    def get_next_neighbour(self, *args, **kwargs):
        global next_neighbour_requested
        global current_image_list
        global image_is_set
        if not solution_applied:
            boolean, next_neighbours = back_end.get_next_neighbour(click_label.text)
            if boolean:
                image_is_set = False
                current_image_list = []
                next_neighbour_requested = True
        else:
            build_meta_fragment()

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
    def apply_resize(self):
        global pl_solution
        global key_fragment_id
        global key_image
        global path_dic
        print(pl_solution)
        for image in current_image_list:
            image_id = image.get_id()
            positions = np.array(image.get_position_memory())
            # positions = current_positions[image_id]
            position = np.array([positions[0], positions[1]])

            image.update(position, 0)

    @mainthread
    def apply_solution(self):
        global pl_solution
        global key_fragment_id
        global key_image
        global path_dic
        center_x = (Window.size[0] / 2)  # Calculate the center of the window in x-axis
        center_y = (Window.size[1] / 2)  # Calculate the center of the window in y-axis

        for image in current_image_list:
            image.remove_score()
            image_id = image.get_id()

            if image_id in pl_solution:
                positions = pl_solution[image_id]
                if path_dic['dataset_name'] == 'RePair_group_28':
                    position = np.array([positions[1], (-1 * positions[0])])  # fix the coordinates
                elif path_dic['dataset_name'] == 'Dafne_group_1':
                    position = np.array([-1 * positions[0] / 2, (positions[1] / 2)])  # fix the coordinates
                else:
                    position = np.array([-1 * positions[0] / 2, (positions[1] / 2)])  # fix the coordinates
                new_positions = np.array([position[0] + center_x, position[1] + center_y])

                r = np.array([image.texture_size[0] / image.norm_image_size[0],
                              image.texture_size[1] / image.norm_image_size[1]])
                # image.rotate(solved_rotation/2)
                image.update(new_positions, positions[2])

    def checking_clock(self, *args, **kwargs):
        global selected_pic
        global anchor_showed
        global neighbour_showed
        global initial_image_updates
        global solution_applied
        global pl_solution
        global next_neighbour_requested
        communicate_thread_lock.acquire()
        self.toolbar_changes(toolbar_color)

        clicked = self.clicked
        if (clicked.isdigit()) & (selected_pic == 0):
            self.widget_list[4].disabled = False
            selected_pic = int(clicked)
        if clicked.isdigit():
            selected_pic = int(clicked)
        if not anchor_showed:
            if (back_end.get_select_anchor_done()) & (len(current_image_list) == 0):
                self.set_images(1)
                self.show_images(self)
                anchor_showed = True
        elif ((back_end.get_select_anchor_done()) & (len(current_image_list) == 0) &
              (back_end.get_select_neighbour_done()) & (not neighbour_showed)):
            self.set_images(1)
            self.show_images(self)
            neighbour_showed = True
        elif ((back_end.get_select_anchor_done()) & (len(current_image_list) == 0) &
              (back_end.get_select_neighbour_done()) & neighbour_showed & next_neighbour_requested):
            print("HEREERE")
            self.set_images(1)
            self.show_images(self)
            next_neighbour_requested = False
        elif ((back_end.get_select_anchor_done()) & (back_end.get_select_neighbour_done()) & neighbour_showed &
              back_end.get_pl_solver_done() & (not solution_applied)):
            pl_solution = back_end.get_pl_solution()
            self.apply_solution()
            solution_applied = True
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
    back_end.start_anchor_thread()


def start_select_neighbour(self):
    global current_image_list
    global image_is_set
    global key_fragment_id

    back_end.key_fragment = key_fragment_id
    image_is_set = False
    current_image_list = []
    back_end.start_neighbour_thread()


def start_pl_solver(self):
    global current_image_list
    global image_is_set
    global neighbour_ids

    for i in range(0, len(current_image_list)):
        if not (current_image_list[i].get_id() == key_image.get_id()):
            neighbour_ids.append(current_image_list[i].get_id())
    back_end.neighbour_ids = neighbour_ids
    # image_is_set = False
    # current_image_list = []
    back_end.start_pl_solver_thread()


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

        select_anchor_running = back_end.get_select_anchor_running()
        select_neighbour_running = back_end.get_select_neighbour_running()
        pl_solver_running = back_end.get_pl_solver_running()
        select_neighbour_done = back_end.get_select_neighbour_done()
        select_anchor_done = back_end.get_select_anchor_done()
        pl_solver_done = back_end.get_pl_solver_done()

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


def map_mouse_pos_pixel(image_pos, image_pixel, image_size, mouse_pos, width_height, ratio_x, ratio_y, angle):
    global ratio
    ratio = (ratio_x, ratio_y)

    angle_rad = math.radians(angle)

    relative_pos = (mouse_pos[0] - (image_pos[0] + width_height[0]),
                    mouse_pos[1] - (image_pos[1] + width_height[1]))

    rotated_x = (relative_pos[0] * math.cos(-angle_rad)) - (relative_pos[1] * math.sin(-angle_rad))
    rotated_y = (relative_pos[0] * math.sin(-angle_rad)) + (relative_pos[1] * math.cos(-angle_rad))

    reality_pixel = (rotated_x * ratio[0], rotated_y * ratio[1])

    return reality_pixel


def image_reader(image_number, score, has_score):
    name = file_names[image_number]
    click_label.color = (1, 0, 1, 1)
    image = MovableImage(image_path, mask_path, click_label, score, image_number, has_score, name, 0)

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


def setting(): # unified path setting
    global image_path
    global mask_path
    global backend_path
    global path_dic
    global rotation_interval
    comp_path = ""
    pieces_path = ""
    comp_folder = ""
    comp_name = ""
    ground_truth = ""
    dataset_name = ""
    cache_path = ""
    apply_gt = False
    number_of_neighbours = 3
    number_of_anchors = 4
    setting_dir = os.path.join(os.getcwd(), "GUI")
    setting_path = os.path.join(setting_dir, "setting.txt")
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
                                     "dataset_name: RePair_group_28"
                                     ])
    os_path = os.getcwd()
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
    cache_path = "/GUI/pieces/"
    cache_path = os_path + cache_path
    path_dic = {'image_path': image_path, 'mask_path': mask_path, 'backend_path': backend_path, 'comp_path': comp_path,
                'pieces_path': pieces_path, 'comp_folder': comp_folder, 'comp_name': comp_name,
                'rotation_intervals': rotation_intervals, 'ground_truth': ground_truth,
                'number_of_neighbours': number_of_neighbours, 'comp_format': comp_format,
                'apply_gt': apply_gt, 'parameters': parameters, 'number_of_anchors': number_of_anchors,
                'dataset_name': dataset_name, 'cache_path': cache_path}
    image_path = image_path
    mask_path = mask_path
    backend_path = backend_path
    rotation_interval = float(rotation_intervals) / 2

    copy_to_cache()

    back_end.set_backend_path(path_dic)


def copy_to_cache(file_extension = '.png'):
    global path_dic
    source = path_dic['pieces_path']
    cache = path_dic['cache_path']
    # Remove the cache folder and its contents if it exists
    if os.path.exists(cache):
        shutil.rmtree(cache)
        print(f"Cache folder {cache} cleared.")

    # Recreate an empty cache folder
    os.makedirs(cache)
    print(f"Cache folder {cache} recreated.")
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
        print(img)
        return None

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

    b, g, r = cv2.split(src)

    rgba = [b, g, r, alpha]

    dst = cv2.merge(rgba, 4)

    return dst


def read_ground_truth():
    global path_dic
    ground_truth = path_dic['ground_truth']
    if ground_truth != "":
        with open(ground_truth, 'r') as file:
            lines = file.readlines()


def build_meta_fragment():
    global current_image_list, path_dic
    cache = path_dic['cache_path']
    # for image in current_image_list:
    #     print(image.get_position_memory)
    #     remove_image_from_cache(image.get_id())

    canvas_width = 10000
    canvas_height = 10000
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)  # RGBA canvas

    for image in current_image_list:
        image_id = image.get_id()
        path = os.path.join(cache, image_id)

        # Load the image from the cache
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to load image: {image_id}")
                continue

            # Get image's position and rotation
            x, y, rotation = image.get_position_memory()

            # Ensure x and y are integers for slicing
            x = int(x)
            y = int(y)

            # Apply rotation (rotate around the center of the image)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)  # Rotation matrix
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_TRANSPARENT)

            # Ensure image stays within canvas boundaries
            img_h, img_w = rotated_img.shape[:2]
            if x + img_w > canvas_width or y + img_h > canvas_height or x < 0 or y < 0:
                print(f"Skipping image {image_id} due to out-of-bounds placement")
                continue

            # Place rotated image on the canvas at position (x, y)
            canvas[y:y + img_h, x:x + img_w] = rotated_img

            # Remove the image from cache after placing it on the canvas
            remove_image_from_cache(image_id)
        else:
            print(f"Image {image_id} does not exist in the cache.")

        # Save the final meta fragment as a PNG
    output_path = cache + "/meta_fragment.png"
    cv2.imwrite(output_path, canvas)

    print(f"Meta fragment saved at {output_path}")


def remove_image_from_cache(image_id):
    global path_dic
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


if __name__ == '__main__':
    setting()
    Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
    test_thread = threading.Thread(target=communicate_thread, daemon=True)

    test_thread_started = True
    test_thread.start()

    read_ground_truth()

    # transparent_path = os.getcwd() + "/GUI/pieces/"
    # transparent_data(transparent_path)

    # erode_data()

    app = GUIApp().run()


