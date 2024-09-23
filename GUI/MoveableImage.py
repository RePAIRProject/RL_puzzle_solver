import math

from kivy.graphics import Rotate, PopMatrix, PushMatrix, Translate, Scale

from kivy.uix.gridlayout import GridLayout

from kivy.uix.image import Image
from kivy.uix.label import Label

from point_Inside import is_inside_sm
from os import getcwd
import numpy as np

import cv2
from PuzzlePiece import PuzzlePiece


class MovableImage(Image):
    def __init__(self, path, path_bw, label, score, image_number, has_score, name, angle):
        super(MovableImage, self).__init__()

        self.name = name
        os_path = getcwd()
        self.path = path + self.name
        self.path_bw = path_bw + self.name
        self.limit_image = self
        self.source = self.path
        self.drag_timeout = 10000000
        self.drag_distance = 0
        self.image_number = image_number
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        self.label = label
        self.score = score

        self.grid = GridLayout()
        self.grid.cols = 1
        self.grid.rows = 2
        self.angle = self.normalize_angle(angle)

        self.grid.add_widget(self)
        self.has_score = has_score

        self.puzzle_piece = PuzzlePiece(self.name)

        self.score_label = Label()
        self.backend_path = getcwd() + "/GUI/DataBase/Images/RePAIR_plaque_2/"

        self.add_score()
        self.image_bw = self.extract_cv_image()
        self.contours = self.extract_border_polygons()

        self.rot = Rotate()
        self.rot.angle = self.angle
        self.rot.origin = self.center

        self.trans = Translate(0, 0)
        self.trans_bank = (self.trans.x, self.trans.y)

        self.total_delta_x = 0
        self.total_delta_y = 0

        self.position_memory = (self.x, self.y, self.angle)

        self.ratio = np.array([self.texture_size[0] / self.norm_image_size[0],
                               self.texture_size[1] / self.norm_image_size[1]])

        with self.canvas.before:
            PushMatrix()
            # self.canvas.before.add(PushMatrix())
            self.trans = Translate(0, 0)
            self.rot = Rotate(self.rot.origin, self.rot.angle)

        with self.canvas.after:
            PopMatrix()

    def get_id(self, *args, **kwargs):
        return self.name

    def check_mask(self, point):
        try:
            check = self.image_bw[-1 * int(point[1]), int(point[0])]
            # The real image and the coordinate system of here are not really matching... /todo
            if check == 0:
                return False
            return True
        except IndexError:
            return False

        # b, g, r = (self.image_bw[point[0], point[1]])
        # if (b == 0) and (g == 0) and (r == 0):
        #     return False
        # return True

    def extract_cv_image(self):
        image_bw = cv2.imread(self.path_bw, cv2.IMREAD_GRAYSCALE)
        # cv2 image for pixel
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
        print(self.score_label.text)

        self.score_label.text = ""
        print(self.score_label.text)

    def rotate(self, delta_angle):
        self.rot.origin = self.center
        self.rot.angle += delta_angle
        self.rot.axis = (0, 0, 1)
        # self.canvas.before.add(self.rot)
        self.angle = self.angle + delta_angle
        self.angle = self.normalize_angle(self.angle)
        self.set_position_memory(self.position_memory[0], self.position_memory[1], self.angle)

    def translate(self, x, y):
        self.trans.x = x - self.pos[0]
        self.trans.y = y - self.pos[1]
        self.trans_bank = (self.trans.x, self.trans.y)

        self.set_position_memory(x * self.ratio[0], y * self.ratio[1], self.position_memory[2])

    def update_translate(self):
        self.trans.x = self.trans_bank[0]
        self.trans.y = self.trans_bank[1]

    def get_real_pos(self):
        real_pos = (self.pos[0] + self.trans.x, self.pos[1] + self.trans.y)
        return real_pos

    def get_angel(self):
        return self.angle

    def collides(self, x, y):
        return self.collide_point(x - self.trans.x, y - self.trans.y)
        # return self.x <= x <= self.right and self.y <= y - self.trans.y <= self.top

    def update(self, position, solved_rotation, *args, **kwargs):
        self.ratio = np.array([self.texture_size[0] / self.norm_image_size[0],
                               self.texture_size[1] / self.norm_image_size[1]])
        x = position[0] / self.ratio[0]
        y = position[1] / self.ratio[1]
        # x = position[0]
        # y = position[1]
        # self.rotate(solved_rotation/2)

        self.rotate(solved_rotation)
        self.translate(x, y)

    def set_position_memory(self, x, y, angle):
        self.position_memory = (x, y, angle)

    def get_position_memory(self):
        return self.position_memory

    @staticmethod
    def normalize_angle(angle):
        angle = (angle % 360 + 360) % 360
        return angle

    def set_position(self, x, y):
        self.x = x
        self.y = y
        # self.center = self.x + (self.width / 2, self.height / 2)

    def move(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y

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
