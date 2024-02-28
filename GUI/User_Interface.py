import os

from kivy import Config
from kivy.clock import Clock
from kivymd.app import MDApp

from kivy.graphics import Rotate, PopMatrix, PushMatrix
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.behaviors import DragBehavior
from kivy.uix.scatter import Scatter
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar

import Back_End
from select_anchor_RePAIR import get_backend_path

import json
import threading
import time

from MoveableImage import MovableImage

# class GUI(Widget):
#     pass
image_numbers = []
file_names = []
scores = []
Window.clearcolor = (0, 0, 0, 0)


# Window.fullscreen = 'auto'


# Window.borderless = True
# Window.background_color = (1, 1, 1)


def json_reader(file):
    with open(file) as f:
        file_list = f.read()
    parsed_json = json.loads(file_list)

    for i in range(0, len(parsed_json)):
        counter = 0
        word_temp1 = ""
        word_temp2 = ""

        # st.write(json.dumps(parsed_json[i]))
        temp = json.dumps(parsed_json[i])
        for j in range(0, len(temp)):
            # st.write(temp[j])
            if temp[j] == '"':
                counter += 1

            if counter == 3:
                if temp[j] != '"':
                    word_temp1 += temp[j]
            elif counter == 6:
                if (temp[j] != '"') and (temp[j] != ':') and (temp[j] != '}') and (temp[j] != ' '):
                    word_temp2 += temp[j]
        score = float(word_temp2)
        image_numbers.append(i)
        file_names.append(word_temp1)
        scores.append(score)


# backEnd_path = "Images/RePAIR_plaque_2/top10/"
backend_path = ""


def image_reader(image_number, score):

    source = backend_path + "top10/" + file_names[image_number]
    click_label.color = (1, 0, 1, 1)
    image = MovableImage(source, click_label, score, image_number)
    # image.source = source
    return image


buttons = []


class Image(Image):
    def __init__(self):
        super().__init__()
        self.image_number = None

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            click_label.text = str(self.image_number + 1)


click_label = Label()

communication_freq = 0.25  # in seconds


class MainLayout(GridLayout):
    def __init__(self):
        super().__init__()

    # def on_touch_down(self, touch):
    #     # print('Released split1_bottom bar')
    #     print('Y value = %d' % touch.y)
    #     print('X value = %d' % touch.x)
    #     # self.label.text = str(self.image_number + 1)

    the_list = []


class GUIApp(MDApp):
    widget_list = []

    def build(self):
        the_app = self
        the_layout = MDBoxLayout(md_bg_color=(0, 0, 0, 1))
        main_layout = MainLayout()
        main_layout.cols = 1

        main_layout.rows = 3

        toolbar = MDTopAppBar()
        toolbar.orientation = "horizontal"

        main_layout.add_widget(toolbar)

        run_button = Button(text="Run")
        run_button.bind(on_press=start_select_anchor)

        run_button.size_hint_x = 0.5

        show_button = Button(text="Show")
        show_button.bind(on_press=self.show_images)

        show_button.size_hint_x = 0.5
        # show_button.bind(on_press=self)

        # toolbar_layout = GridLayout()
        # toolbar_layout.cols = 1
        # toolbar_layout.rows = 2
        # toolbar_layout.add_widget(run_button)
        # toolbar_layout.add_widget(show_button)
        # toolbar.add_widget(toolbar_layout)

        toolbar.left_action_items.append(["menu", lambda x: the_app.callback()])

        toolbar.add_widget(run_button)
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
        the_layout.add_widget(main_layout)
        self.widget_list.append(toolbar)  # 0 toolbar
        self.widget_list.append(run_button)  # 1 run_button
        self.widget_list.append(show_button)  # 2 show_button
        self.widget_list.append(grid_layout)  # 3 image_view
        Clock.schedule_interval(self.checking_clock, communication_freq)  # Graphic Internal Thread to communicate
        return the_layout

    global select_anchor_running
    global toolbar_color
    toolbar_bg = 0
    global backend_path

    def show_images(self, *args, **kwargs):
        global backend_path
        current_path = os.getcwd() + "/GUI/"
        backend_path = get_backend_path()
        path = current_path + "top_10_fragments.json"
        json_reader(path)

        for i in range(len(image_numbers)):
            score_label = Label()
            buttons.append(score_label)
            score_label.color = (1, 0, 0, 1)
            score_label.text = str(scores[i])

            image = image_reader(i, scores[i])

            self.widget_list[3].add_widget(image.get_grid())
        print("show")

    def checking_clock(self, *args, **kwargs):

        communicate_thread_lock.acquire()
        self.toolbar_changes(toolbar_color)
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
    Back_End.start_back_end()


select_anchor_running = None

toolbar_color = 0  # 0 for light blue, -1 for red, 1 for green


def communicate_thread():  # communication thread, to communicate between UI, Graphic and BackEnd
    global select_anchor_running
    print("Test thread")
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
        if select_anchor_running is not None:
            if select_anchor_running:
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


if __name__ == '__main__':
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




