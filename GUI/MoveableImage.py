from kivy import Config
from kivy.app import App
from kivy.graphics import Rotate, PopMatrix, PushMatrix, Translate, Scale
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
from kivymd.uix.behaviors import RotateBehavior

from point_Inside import is_inside_sm
from os import getcwd

import cv2


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


class MovableImage(Image, RotateBehavior):
    def __init__(self, source, label, score, image_number, has_score, path):
        super(MovableImage, self).__init__()
        RotateBehavior.__init__(self)

        self.name = path
        os_path = getcwd()
        self.path = os_path + "/GUI/Images/RePAIR_plaque_2/RGBA_merged/" + self.name
        self.path_bw = os_path + "/GUI/Images/RePAIR_plaque_2/FG_merged/" + self.name
        self.limit_image = self
        self.source = source
        self.drag_timeout = 10000000
        self.drag_distance = 0
        self.image_number = image_number
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        self.label = label
        self.score = score

        self.grid = GridLayout()
        self.grid.cols = 1
        self.grid.rows = 2

        self.grid.add_widget(self)
        self.has_score = has_score

        self.score_label = Label()
        self.backend_path = getcwd() + "/GUI/Images/RePAIR_plaque_2/"

        self.add_score()
        self.image_bw = self.extract_cv_image()
        self.contours = self.extract_border_polygons()

        # self.update()
        # opencv binary pic

    def check_mask(self, point):
        self.rotate(45)
        check = self.image_bw[int(point[0]), int(point[1])]
        if check == 0:
            return False
        return True

        # b, g, r = (self.image_bw[point[0], point[1]])
        # if (b == 0) and (g == 0) and (r == 0):
        #     return False
        # return True

    def extract_cv_image(self):
        image_bw = cv2.imread(self.path_bw, cv2.IMREAD_GRAYSCALE)
        return image_bw

    def extract_border_polygons(self):
        contours, heir = cv2.findContours(self.image_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        coords = []
        for j in range(len(contours)):
            for i in range(len(contours[j])):
                point = (contours[j][i][0][0], contours[j][i][0][1])
                coords.append(point)
        return contours
    # /todo down sampling

    def check_if_inside(self, point):
        for j in range(len(self.contours)):
            coords = []
            for i in range(len(self.contours[j])):
                point = (self.contours[j][i][0][0], self.contours[j][i][0][1])
                coords.append(point)
            if is_inside_sm(coords, point):
                return True
        return False

    def add_score(self, *args, **kwargs):
        if self.has_score:
            self.score_label.text = str(self.score)
            self.grid.add_widget(self.score_label)

    def remove_score(self):
        self.grid.remove(self.score_label)

    def rotate(self, delta_angle):
        self.canvas.before.add(PushMatrix())
        self.canvas.before.add(Rotate(angle=delta_angle, origin=self.center))

        self.canvas.after.add(PopMatrix())

    def move(self):
        pass

    def get_scatter(self):
        return self.scatter

    def get_number(self):
        return self.image_number

    def get_grid(self):
        return self.grid

    def get_name(self):
        return self.name

    def get_path(self):
        return self.path
