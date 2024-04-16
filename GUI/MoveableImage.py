import math

from kivy.graphics import Rotate, PopMatrix, PushMatrix, Translate, Scale

from kivy.uix.gridlayout import GridLayout

from kivy.uix.image import Image
from kivy.uix.label import Label

from point_Inside import is_inside_sm
from os import getcwd

import cv2


class MovableImage(Image):
    def __init__(self, source, label, score, image_number, has_score, path, angle):
        super(MovableImage, self).__init__()

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
        self.angle = self.normalize_angle(angle)

        self.grid.add_widget(self)
        self.has_score = has_score

        self.score_label = Label()
        self.backend_path = getcwd() + "/GUI/Images/RePAIR_plaque_2/"

        self.add_score()
        self.image_bw = self.extract_cv_image()
        self.contours = self.extract_border_polygons()

        self.rot = Rotate()
        self.rot.angle = self.angle
        self.rot.origin = self.center

        self.trans = Translate(0, 0)

        self.total_delta_x = 0
        self.total_delta_y = 0

        self.vx = self.x
        self.vy = self.y

        # self.bind(pos=self.binding)

        # self.update()
        # opencv binary pic

    def update_virtual_pos(self, *args, **kwargs):
        self.vx = self.x
        self.vy = self.y
        print(self.vx)
        print(self.vy)
        # self.center = (self.x + self.width / 2, self.y + self.height)

    def check_mask(self, point):
        try:
            check = self.image_bw[int(point[0]), int(point[1])]
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
        with self.canvas:
            PushMatrix()
            # self.canvas.before.add(PushMatrix())
            self.rot = Rotate(self.rot.origin, self.rot.angle)
            self.rot.origin = self.center
            self.rot.angle = delta_angle
            self.rot.axis = (0, 0, 1)
            # self.canvas.before.add(self.rot)
            self.angle = self.angle+delta_angle
            self.angle = self.normalize_angle(self.angle)
            print(self.angle)
        with self.canvas.after:
            PopMatrix()

    def translate(self, x, y):
        # Calculate the displacement from the object's current center to the target position
        dx = x - self.x
        dy = y - self.y

        # Convert the rotation angle to radians
        angle_rad = math.radians(self.angle)

        print(angle_rad)

        # Rotate the translation vector (dx, dy) based on the current rotation angle
        new_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        new_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

        # # Scale the translation vector to compensate for the rotation
        # scale_factor = math.cos(angle_rad)
        # new_dx *= scale_factor
        # new_dy *= scale_factor

        new_dx = round(new_dx, 10)
        new_dy = round(new_dy, 10)

        # Update the object's center position
        self.x += new_dx
        self.y += new_dy
        # self.rot.origin = self.center
        # self.rot.origin = self.center

        print("x", dx, "y", dy)
        print("new_dx", new_dx, "new_dy")

    @staticmethod
    def normalize_angle(angle):
        angle = (angle % 360 + 360) % 360
        return angle

    def set_position(self, x, y):
        self.x = x
        self.y = y
        # self.center = self.x + (self.width / 2, self.height / 2)

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
