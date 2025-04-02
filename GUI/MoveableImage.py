import math

from kivy.graphics import Rotate, PopMatrix, PushMatrix, Translate, Scale
from kivy.graphics.transformation import Matrix
from kivy.multistroke import bounding_box
from kivymd.uix.behaviors import ScaleBehavior

from kivy.uix.gridlayout import GridLayout

from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics import Color
from shapely.affinity import rotate

from enum import Enum

from GUI.point_Inside import is_inside_sm
from os import getcwd
import numpy as np

import cv2
from GUI.PuzzlePiece import PuzzlePiece

class Status(Enum):
    ACCEPTED = "accepted"
    DENIED = "denied"
    NEUTRAL = "neutral"


class MovableImage(Image):
    def __init__(self, path, path_bw, label, score, image_number, has_score, name, angle, is_anchor=False, **kwargs):
        super(MovableImage, self).__init__()
        self.offset = [0, 0]
        self.is_selected = False
        self.name = name
        os_path = getcwd()
        self.path = path + self.name
        self.base_scale_factor = 1.0
        self.true_scale = 1.0
        self.scale_factor = self.base_scale_factor
        self.zoom_scale = Scale(x=self.scale_factor, y=self.scale_factor, origin=(0, 0))
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

        self.mask = self.get_mask()

        self.rot = Rotate()
        self.rot.angle = self.angle
        self.rot.origin = self.center

        self.trans = Translate(0, 0)
        self.trans_bank = (self.trans.x, self.trans.y)

        self.total_delta_x = 0
        self.total_delta_y = 0

        self.position_memory = (self.x, self.y, self.angle)

        self.ratio = np.array([1, 1])
        self.update_ratio()
        self.is_anchor = is_anchor

        self.is_grabbed = False
        self.is_locked = False
        self.status = Status.NEUTRAL

        self.lock_movement = False

        with self.canvas.before:
            PushMatrix()
            # self.canvas.before.add(PushMatrix())
            self.canvas.before.add(self.zoom_scale)
            self.trans = Translate(0, 0)
            self.rot = Rotate(self.rot.origin, self.rot.angle)

        with self.canvas.after:
            PopMatrix()

        self.zoom_matrix = self.zoom_scale.matrix.tolist()
        self.trans_matrix = self.trans.matrix.tolist()
        self.rotate_matrix = self.rot.matrix.tolist()
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def get_id(self, *args, **kwargs):
        return self.name

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

    def get_mask(self):
        mask = np.zeros(self.image_bw.shape, dtype=np.uint8)
        for j in range(len(self.contours)):
            cv2.drawContours(mask, self.contours, j, 255, -1)
        return mask

    def add_score(self, *args, **kwargs):
        if self.has_score:
            self.score_label.text = str(self.score)
            self.grid.add_widget(self.score_label)

    def remove_score(self):
        self.score_label.text = ""

    def rotate(self, delta_angle):
        self.rotate_point(delta_angle, self.center)
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def rotate_point(self, delta_angle, point):
        self.rot.origin = point

        self.rot.axis = (0, 0, 1)
        # self.canvas.before.add(self.rot)
        self.angle = self.angle + delta_angle
        self.angle = self.normalize_angle(self.angle)
        self.rot.angle = self.angle
        self.position_memory = [self.position_memory[0], self.position_memory[1], self.angle]

    def translate(self, x, y):

        self.trans.x = x - self.pos[0]
        self.trans.y = y - self.pos[1]
        self.trans_bank = (self.trans.x, self.trans.y)
        self.position_memory = [x * self.ratio[0], y * self.ratio[1], self.position_memory[2]]
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def update_translate(self):
        self.trans.x = self.trans_bank[0]
        self.trans.y = self.trans_bank[1]

    def get_real_pos(self):
        real_pos = (self.pos[0] + self.trans.x, self.pos[1] + self.trans.y)
        return real_pos

    def get_angel(self):
        return self.angle

    def update_positions(self, position, solved_rotation, *args, **kwargs):
        self.update_ratio()
        x = position[0] / self.ratio[0]
        y = position[1] / self.ratio[1]

        # x = position[0]
        # y = position[1]
        # self.rotate(solved_rotation/2)

        self.rotate(solved_rotation - self.angle)
        self.translate(x, y)

    def update_ratio(self):
        self.ratio = np.array([self.texture_size[0] / self.norm_image_size[0],
                               self.texture_size[1] / self.norm_image_size[1]])

    def zoom_default(self, factor, origin):
        self.zoom_scale.origin = origin
        self.scale_factor = factor
        self.zoom_scale.x = factor
        self.zoom_scale.y = factor
        self.base_scale_factor = factor
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def zoom_at_point(self, factor, origin):
        new_scale = self.scale_factor * factor
        self.scale_factor = new_scale

        last_origin = self.zoom_scale.origin
        origin_diff = (origin[0] - last_origin[0], origin[1] - last_origin[1])

        if (self.scale_factor < self.true_scale < new_scale) or (
                self.scale_factor > self.true_scale > new_scale):
            new_scale = self.true_scale

        if (new_scale > 0.95) & (new_scale < 1.05):
            new_scale = 1.0
        if new_scale < 0.5:
            new_scale = 0.5
        elif new_scale > 10.0:
            new_scale = 10.0
        origin_diff = (origin_diff[0] / self.scale_factor, origin_diff[1] / self.scale_factor)

        self.zoom_scale.origin = (last_origin[0] + origin_diff[0], last_origin[1] + origin_diff[1])

        # Adjust the scale factor
        self.scale_factor = new_scale
        self.zoom_scale.x = self.scale_factor
        self.zoom_scale.y = self.scale_factor
        self.base_scale_factor = self.scale_factor
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def zoom_reset(self):
        self.scale_factor = 1.0
        self.base_scale_factor = 1.0
        self.zoom_scale.x = 1.0
        self.zoom_scale.y = 1.0

    def calculate_transform_matrix(self):
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix.multiply(self.rot.matrix))

    def collides(self, mouse_pos):
        relative_pos = mouse_pos

        relative_pos = self.transform_matrix.inverse().transform_point(relative_pos[0], relative_pos[1], 0)

        x, y = relative_pos[:2]

        return self.x <= x <= self.right and self.y <= y <= self.top

    def map_mouse_pos_pixel(self, mouse_pos):
        width_height = ((self.width - self.norm_image_size[0]) / 2,
                        (self.height - self.norm_image_size[1]) / 2)
        image_pos = (self.pos[0], self.pos[1])
        relative_translation = (image_pos[0] + width_height[0], image_pos[1] + width_height[1])

        relative_translation = Matrix().translate(relative_translation[0], relative_translation[1], 0)

        ratio_x = self.texture_size[0] / self.norm_image_size[0]
        ratio_y = self.texture_size[1] / self.norm_image_size[1]

        relative_pos = mouse_pos

        relative_pos = self.transform_matrix.inverse().transform_point(relative_pos[0], relative_pos[1], 0)
        relative_pos = relative_translation.inverse().transform_point(relative_pos[0], relative_pos[1], 0)

        reality_pixel = (relative_pos[0] * ratio_x, relative_pos[1] * ratio_y)

        return reality_pixel

    def check_mask(self, point):
        try:
            check = self.image_bw[-1 * int(point[1]), int(point[0])]
            if check == 0 or check is None:
                return False
            return True
        except IndexError:
            return False

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

    def select(self):
        self.is_selected = True
        self.set_color((0.8, 0.8, 1, 1))  # blue

    def deselect(self):
        self.is_selected = False
        if self.is_anchor:
            self.set_color((0.6, 1, 0.6, 1))
        else:
            self.set_color((1, 1, 1, 1))

    def set_probability(self, probability):
        probability = np.round(probability * 100)
        if probability <25:
            self.set_color((1, 1, 1, 0.25)) # transparent
        else:
            alpha = probability / 100
            if self.is_anchor:
                self.set_color((0.6, 1, 0.6, alpha))
            else:
                self.set_color((1, 1, 1, alpha)) # gradually transparent to original

    def select_toggle(self):
        if self.is_selected:
            self.deselect()
        else:
            self.select()

    def set_color(self, color):
        if not self.is_grabbed:
            self.color = color
        else:
            self.color = (0.8, 0.8, 1, 1)

    def update_offset(self, center):
        offset = [(center[0] - self.parent.size[0] / 2 - (self.parent.pos[0] - self.pos[0]) / 2) * self.ratio[0],
                  (center[1] - self.parent.size[1] / 2 - (self.parent.pos[1] - self.pos[1]) / 2) * self.ratio[1]]
        return offset

    def set_is_grabbed(self, is_grabbed):
        self.is_grabbed = is_grabbed

    def set_anchor(self, value):
        if value:
            self.is_anchor = True
            self.set_color((0.6, 1, 0.6, 1))
        else:
            self.is_anchor = False
            self.set_color((1, 1, 1, 1))

    def extract_bounding_box(self):
        """
        Computes the bounding box in Kivy coordinates after applying all transformations,
        including the relative position of the widget.
        """
        if not self.contours:
            return [0, 0, 0, 0]

        transformed_points = []

        # Compute relative translation matrix (same logic as in map_mouse_pos_pixel)
        width_height = ((self.width - self.norm_image_size[0]) / 2,
                        (self.height - self.norm_image_size[1]) / 2)
        image_pos = (self.pos[0], self.pos[1])
        relative_translation = Matrix().translate(image_pos[0] + width_height[0], image_pos[1] + width_height[1], 0)

        # Apply transformations to each contour point
        for contour in self.contours:
            for point in contour:
                x, y = point[0][0], point[0][1]  # Extract contour point

                # First, apply full transformation matrix
                total_transform = self.transform_matrix.multiply(relative_translation)
                transformed_point = total_transform.transform_point(x, y, 0)
                # transformed_point = self.transform_matrix.inverse().transform_point(transformed_point[0], transformed_point[1], 0)

                # # Then apply relative translation matrix to get final Kivy coordinates
                # kivy_point = relative_translation.transform_point(transformed_point[0], transformed_point[1], 0)
                transformed_points.append(transformed_point[:2])  # Extract (x, y) only

        if not transformed_points:
            return [0, 0, 0, 0]

        # Convert to numpy for easy min/max computation
        transformed_points = np.array(transformed_points)

        # Compute bounding box in Kivy coordinate system
        min_x, max_x = np.min(transformed_points[:, 0]), np.max(transformed_points[:, 0])
        min_y, max_y = np.min(transformed_points[:, 1]), np.max(transformed_points[:, 1])

        edge_x = max_x - min_x
        edge_y = max_y - min_y

        x_offset = -1 * edge_x * 15/100
        y_offset = -1 * edge_y * 15/100

        min_x = min_x - x_offset
        max_x = max_x + x_offset
        min_y = min_y - y_offset
        max_y = max_y + y_offset



        return [min_x, min_y, max_x, max_y]
