from kivy import Config
from kivy.app import App
from kivy.graphics import Rotate, PopMatrix, PushMatrix
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix. behaviors import DragBehavior


import json

from MoveableImage import MovableImage

# class GUI(Widget):
#     pass
image_numbers = []
file_names = []
scores = []
Window.clearcolor = (0, 0, 0)
# Window.fullscreen = 'auto'


Window.borderless = True
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


def image_reader(image_number):
    source = "Images/top10/" + file_names[image_number]
    image = MovableImage(click_label)
    image.source = source
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


class GUIApp(App):
    def build(self):

        main_layout = GridLayout()
        main_layout.cols = 1
        main_layout.rows = 2
        
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
        for i in range(len(image_numbers)):
            inside_layout = GridLayout()
            inside_layout.cols = 1
            inside_layout.rows = 2
            score_label = Label()
            buttons.append(score_label)
            score_label.text = str(scores[i])

            image = image_reader(i)
            image.image_number = i

            inside_layout.add_widget(image)
            inside_layout.add_widget(score_label)

            grid_layout.add_widget(inside_layout)

        # for i in range(10):
        #     button = Button()
        #     button.text = str(i + 1)
        #     button.height = 1000
        #     button.width = 1000
        #
        #     grid_layout.add_widget(button)
        # grid_layout.add_widget(button)
        # grid_layout.add_widget(widget)
        return main_layout


if __name__ == '__main__':
    path = "Images/top10/top_10_fragments.json"
    json_reader(path)
    GUIApp().run()
