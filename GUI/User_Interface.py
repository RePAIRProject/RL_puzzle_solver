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
from kivy.uix.behaviors import DragBehavior
from kivy.uix.scatter import Scatter
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar
import matplotlib.path as mpltPath
from shapely.geometry import Point, Polygon
from point_Inside import is_inside_sm


from kivy.modules.inspector import Inspector
from kivy.core.window import Window
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.factory import Factory

import Back_End
from select_anchor_RePAIR import get_backend_path

import json
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


def image_reader(image_number, score, has_score):
    path = file_names[image_number]
    source = backend_path + "RGBA_merged/" + path
    click_label.color = (1, 0, 1, 1)
    image = MovableImage(source, click_label, score, image_number, has_score, path)
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

        Window.bind(on_motion=self.on_touch_move, on_key_down=self._on_keyboard_down)

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

        # show_button.bind(on_press=self)

        # toolbar_layout = GridLayout()
        # toolbar_layout.cols = 1
        # toolbar_layout.rows = 2
        # toolbar_layout.add_widget(anchor_button)
        # toolbar_layout.add_widget(show_button)
        # toolbar.add_widget(toolbar_layout)

        toolbar.left_action_items.append(["menu", lambda x: the_app.callback()])

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
        scores = Back_End.image_scores
        current_image_list = []
        # score_label.text = str(scores[i])
        backend_path = get_backend_path()
        file_names = Back_End.image_names
        for i in range(len(Back_End.image_numbers)):
            image = image_reader(i, scores[i], has_score)
            print(file_names[i])
            # image.fit_mode = "contain"
            current_image_list.append(image)

    @mainthread
    def on_touch_move(self, window, pos, touch, *args, **kwargs):  # Mouse Listener
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                print('up')
            elif touch.button == 'scrollup':  # they are inverse...
                print('down')
        window_size = Window.size
        mouse_pos = (window_size[0] * touch.spos[0], window_size[1] * touch.spos[1])
        if touch.button == 'left':
            for i in range(len(current_image_list)):
                # print("image " + str(i) + ": " + "x: " + str(current_image_list[i].pos[0]/Window.size[0]) + "    y: " + str(current_image_list[i].pos[1]/Window.size[1]))
                # print(current_image_list[i].texture_size)
                # print(current_image_list[i].get_norm_image_size())
                # print("Xratio: " + str(current_image_list[i].texture_size[0]/current_image_list[i].get_norm_image_size()[0]))
                # print("Yratio: " + str(current_image_list[i].texture_size[1] / current_image_list[i].get_norm_image_size()[1]))
                if current_image_list[i].collide_point(mouse_pos[0], mouse_pos[1]):
                    print(str(i) + ": yes yes")
                    print(current_image_list[i].size)
                    width_height = ((current_image_list[i].width - current_image_list[i].norm_image_size[0]) / 2,
                                    (current_image_list[i].height - current_image_list[i].norm_image_size[1]) / 2)
                    pixel = map_mouse_pos_pixel(current_image_list[i].pos, current_image_list[i].texture_size,
                                                current_image_list[i].get_norm_image_size(), mouse_pos, width_height)
                    check_border(pixel, current_image_list[i].get_path())
        # todo windows to widget mapping
        # print("pos", touch.spos)

    @mainthread
    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):  # Keyboard Listener
        if len(modifiers) > 0:
            print("modifiers", modifiers)
        print(keyboard)
        if len(modifiers) > 0 and modifiers[0] == 'ctrl' and text == 'a':  # Ctrl+a
            print("\nThe key", keycode, "have been pressed")
            print(" - text is %r" % text)
            print(" - modifiers are %r" % modifiers)

    @mainthread
    def mouse_pos(self, window, pos, *args, **kwargs):
        # print("mouse_pos: ", pos)
        # print(pos)
        return True

    @mainthread
    def show_images(self, *args, **kwargs):
        global showed_image_list
        if len(showed_image_list) != 0:
            self.clear_images()
        # current_path = os.getcwd() + "/GUI/"

        # path = current_path + "top_10_fragments.json"
        # json_reader(path)
        showed_image_list = []

        for i in range(len(Back_End.image_numbers)):
            showed_image_list.append(current_image_list[i].get_grid())
            self.widget_list[3].add_widget(showed_image_list[i])
            print("added")

    def clear_images(self, *args, **kwargs):
        global showed_image_list
        for i in range(len(showed_image_list)):
            self.widget_list[3].remove_widget(showed_image_list[i])
        showed_image_list = []

    def checking_clock(self, *args, **kwargs):
        global selected_pic
        global anchor_showed
        global neighbour_showed
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
        elif (Back_End.get_select_anchor_done()) & (len(current_image_list) == 0) & (Back_End.get_select_neighbour_done()) & (not neighbour_showed):
            self.set_images(0)
            self.show_images(self)
            neighbour_showed = True
            print("neighbour")
        # if select_anchor_running is not None:
        #     if select_anchor_running:
        #         self.widget_list[0].md_bg_color = (0.545098039, 0, 0, 1)  # Set Toolbar Red
        #         self.widget_list[1].disabled = True
        #         self.widget_list[2].disabled = True  # commit
        #     else:
        #         self.widget_list[0].md_bg_color = (0.141176471, 0.529411765, 0.129411765, 1)  # Set Toolbar Green
        #         self.widget_list[1].disabled = True
        #         self.widget_list[2].disabled = False  # todo find sth else for this...
        #
        # else:
        #     self.widget_list[0].md_bg_color = (0.678431373, 0.847058824, 0.901960784, 1)  # Set Toolbar Blue
        #
        #     self.widget_list[2].disabled = True
        #
        # if test_thread_started:
        #     print("Test thread started")
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
    image_is_set = False
    current_image_list = []
    Back_End.start_neighbour_thread()


select_anchor_running = None
select_neighbour_running = None

select_anchor_done = None
select_neighbour_done = None

toolbar_color = 0  # 0 for light blue, -1 for red, 1 for green


def communicate_thread():  # communication thread, to communicate between UI, Graphic and BackEnd
    global select_anchor_running
    global select_neighbour_running
    global select_anchor_done
    global select_neighbour_done
    global toolbar_color
    while True:
        communicate_thread_lock.acquire()
        # if not test_thread_started:
        #     print("test_thread")
        #     return
        # print(Back_End.get_select_anchor_running())
        # if not Back_End.get_select_anchor_running():
        #     return

        select_anchor_running = Back_End.get_select_anchor_running()
        select_neighbour_running = Back_End.get_select_neighbour_running()
        select_neighbour_done = Back_End.get_select_neighbour_done()
        select_anchor_done = Back_End.get_select_anchor_done()
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
        # if Back_End.get_select_anchor_running():
        #     print(Back_End.get_select_anchor_running())
        # elif Back_End.get_select_anchor_running():
        #     print(Back_End.get_select_anchor_running())
        time.sleep(communication_freq)  # Thread sleep timer


def check_border(pixel, path, *args, **kwargs):
    pixel_x = pixel[0]
    pixel_y = pixel[1]
    image_bw = select_what_check(path)
    contours, heir = cv2.findContours(image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polygon_list = []
    is_contains = False
    raytracing = False
    for j in range(len(contours)):
        coords = []
        for i in range(len(contours[j])):
            point = (contours[j][i][0][0], contours[j][i][0][1])
            coords.append(point)
        # array = np.array(list(map(float, coords)))
        # print(array)
        # path = mpltPath.Path(contours[j])
        # inside2 = path.contains_points(Point(Point(pixel_x, pixel_y)))
        poly = Polygon(coords)
        polygon_list.append(poly)
        point = [pixel_x, pixel_y]
        raytracing = is_inside_sm(coords, point)
        # if poly.contains(Point(pixel_x, pixel_y)):
        #     is_contains = True
        #     break
        if raytracing:
            is_contains = True
            break
    print(is_contains)

    # todo down-sampling


def select_what_check(path):
    args = get_args()
    images_folder = args.fg_mask

    neighbour_fragments = cv2.imread(args.fg_mask + '/' + path, cv2.IMREAD_GRAYSCALE)
    return neighbour_fragments


def map_mouse_pos_pixel(image_pos, image_pixel, image_size, mouse_pos, width_height):
    ratio = (image_pixel[0]/image_size[0], image_pixel[1]/image_size[1])

    if mouse_pos[0] < 0 or mouse_pos[1] < 0:
        print('clicked outside of image\n')

    relative_pos = (mouse_pos[0] - image_pos[0] - width_height[0], mouse_pos[1] - image_pos[1] - width_height[1])
    # relative_pos = (touch.x - image_pos[0] - width_height[0], touch.y - image_pos[1] - width_height[1])
    print(relative_pos[0])
    print(relative_pos[1])
    reality_pixel = (relative_pos[0]*ratio[0], relative_pos[1]*ratio[1])
    return reality_pixel


def get_args():
    parser = argparse.ArgumentParser(description='Select anchor / key fragment')
    # parser.add_argument('-d', '--dataset', type=str, default='/home/sinem/PycharmProjects/User-Interface-Repair-Project/Images/RePAIR_plaque_2/RGBA_merged', help='data folder')
    parser.add_argument('-d', '--dataset', type=str,
                        default=backend_path + 'RGBA_merged',
                        help='data folder')
    # parser.add_argument('-f', '--fg_mask', type=str, default='/home/sinem/PycharmProjects/User-Interface-Repair-Project/Images/RePAIR_plaque_2/FG_merged', help='data folder')
    parser.add_argument('-f', '--fg_mask', type=str,
                        default=backend_path + 'FG_merged',
                        help='data folder')
    answer = parser.parse_args()
    return answer


if __name__ == '__main__':
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    test_thread = threading.Thread(target=communicate_thread, daemon=True)
    test_thread_started = True
    test_thread.start()
    # sorted_image_scores = select_anchor()
    # print(sorted_image_scores)
    # path = backEnd_path + "top_10_fragments.json"
    # current_path = os.getcwd() + "/GUI/"
    # backEnd_path = get_backend_path()
    # path = current_path + "top_10_fragments.json"
    # json_reader(path)

    app = GUIApp().run()
