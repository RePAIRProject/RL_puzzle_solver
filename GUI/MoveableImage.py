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

global click_label


class MovableImage(DragBehavior, Image):
    def __init__(self, label, **kwargs):
        super(MovableImage, self).__init__(**kwargs)
        self.drag_timeout = 10000000
        self.drag_distance = 0
        self.image_number = None
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        # self.rotate = Rotate(angle)
        self.label = label

        self.canvas.before.add(PushMatrix())
        # self.canvas.before.add(self.rotate)
        self.canvas.after.add(PopMatrix())

        self.bind()
        # self.bind(self.update_canvas)

    # label = Label()

    # def set_label(self, label):
    #     self.label = label

    def update_canvas(self, *args):
        self.rotate.origin = self.center
    # def on_touch_down(self, touch):
    #     if self.collide_point(*touch.pos):
    #         click_label.text = str(self.image_number + 1)

    def on_pos(self, *args):
        print("here")
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        self.label.text = str(self.image_number + 1)
        # click_label.text = str(self.image_number + 1)
        # label = self.image_number + 1

    def on_size(self, *args):
        self.drag_rectangle = [self.x, self.y, self.width, self.height]