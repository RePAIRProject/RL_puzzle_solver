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
from kivy.uix.behaviors import DragBehavior
from kivy.uix.scatter import Scatter
from kivy.uix.relativelayout import RelativeLayout


class MovableLayout(RelativeLayout):
    def __init__(self):
        super(MovableLayout, self).__init__()
        self.size_hint = (None, None)
        # self.size = (64, 64)
        # self.drag_timeout = 10000000
        # self.drag_distance = 0
        # self.drag_rectangle = [self.x, self.y, self.width, self.height]

    # def on_pos(self, *args):
    #     self.drag_rectangle = [self.x, self.y, self.width, self.height]
    #     # self.label.text = str(self.image_number + 1)


score_label = None
path = None


class MovableImage(Image):
    def __init__(self, source, label, score, image_number, has_score, path):
        super(MovableImage, self).__init__()
        global score_label
        self.path = path
        self.limit_image = self
        self.source = source
        self.drag_timeout = 10000000
        self.drag_distance = 0

        self.image_number = image_number
        self.drag_rectangle = [self.x, self.y, self.width, self.height]

        self.label = label
        self.score = score

        self.bind()
        self.scatter = Scatter()
        self.scatter.pos = (0, 0)
        self.scatter.do_rotation = True
        self.scatter.do_scale = False
        self.scatter.do_translate = False
        self.scatter.rotation = 90  # Degree

        self.grid = GridLayout()
        self.grid.cols = 1
        self.grid.rows = 2

        self.grid.add_widget(self)

        if has_score:
            score_label = Label()
            score_label.text = str(self.score)
            self.grid.add_widget(score_label)

    def get_grid(self):
        return self.grid

    def get_path(self):
        return self.path

    def remove_score(self):
        global score_label
        self.grid.remove(score_label)

    def get_scatter(self):
        return self.scatter

    def update_canvas(self, *args):
        self.rotate.origin = self.center

    def add_score(self, *args, **kwargs):
        self.grid.add_widget(self.score_label)

    def on_pos(self, *args):

        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        # self.label.text = str(self.image_number + 1)
        print("here" + str(self.image_number))

    def on_size(self, *args):
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
