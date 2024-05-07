from os import getcwd

from kivy.graphics import PushMatrix
from kivy.graphics import PopMatrix
from kivy.graphics import Rotate
from kivy.graphics import Translate
from kivy.graphics import Rectangle
from kivy.graphics import Scale
from kivy.graphics import Color
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout

# https://github.com/Poikilos/KivySpinningBunnies/blob/master/spritewidget.py


class RotatingImage(Widget):
    freeAngle = None
    freePos = None
    _translate_instruction = None
    _rotate_instruction = None
    _scale_instruction = None
    _rectangle_instruction = None
    _color_instruction = None

    degreesPerSecond = None
    speedPixelsPerFrameXYPair = None

    def __init__(self, path, pos, **kwargs):
        super(RotatingImage, self).__init__(**kwargs)
        self.freeAngle = 0.0
        self.degreesPerSecond = 0.0
        self.freePos = pos
        self.norm_image_size = self.freePos
        self.texture_size = (41.0, 41.0)

        self.name = path
        os_path = getcwd()
        self.path = os_path + "/GUI/Images/RePAIR_plaque_2/RGBA_merged/" + self.name

        self._rectangle_instruction = Rectangle(source=self.path, pos=(0, 0), size=(41.0, 41.0))
        self._rotate_instruction = Rotate(angle=self.freeAngle, origin=(self._rectangle_instruction.size[0] / 2.0, self._rectangle_instruction.size[1] / 2.0))
        self._scale_instruction = Scale(1.0, 1.0, 1.0)
        self._translate_instruction = Translate(0, 0)
        self._color_instruction = Color(Color(1.0, 1.0, 1.0, 1.0))

        self.canvas.add(PushMatrix())
        self.canvas.add(self._translate_instruction)
        self.canvas.add(self._rotate_instruction)
        self.canvas.add(self._scale_instruction)
        self.canvas.add(self._color_instruction)
        self.canvas.add(self._rectangle_instruction)
        self.canvas.add(PopMatrix())

        self.refresh()

    #         with self.canvas.after:
    #             PopMatrix()
    def check_mask(self, *args, **kwargs):
        return True
    def get_norm_image_size(self):
        return self.norm_image_size

    def refresh(self):
        if (self.speedPixelsPerFrameXYPair is not None):
            self.freePos = self.freePos[0] + self.speedPixelsPerFrameXYPair[0], self.freePos[1] + \
                           self.speedPixelsPerFrameXYPair[1]
        self.freeAngle += self.degreesPerSecond
        self._change_instructions()
        self.x = self.freePos[0]
        self.y = self.freePos[1]

    def _change_instructions(self):
        self._translate_instruction.x = self.freePos[0] - self._rectangle_instruction.size[
            0] * self._scale_instruction.x
        self._translate_instruction.y = self.freePos[1] - self._rectangle_instruction.size[
            1] * self._scale_instruction.y
        # self._rectangle_instruction.pos = self.freePos[0]-self._rectangle_instruction.size[0]*self._scale_instruction.x/2.0, self.freePos[1]-self._rectangle_instruction.size[1]*self._scale_instruction.y/2.0
        self._rotate_instruction.origin = self._rectangle_instruction.size[0] * self._scale_instruction.x / 2.0, \
                                          self._rectangle_instruction.size[1] * self._scale_instruction.x / 2.0
        self._rotate_instruction.angle = self.freeAngle

    def get_center_x(self):
        return self.freePos[0]

    def get_center_y(self):
        return self.freePos[1]

    def getCenterPoint(self):
        return self.freePos[0], self.freePos[1]

    def set_center_x(self, x):
        self.freePos = float(x), self.freePos[1]
        self._change_instructions()

    def set_center_y(self, y):
        self.freePos = self.freePos[0], float(y)
        self._change_instructions()

    def set_center_point(self, pos):
        self.freePos = float(pos[0]), self.freePos[1]
        self.freePos = self.freePos[0], float(pos[1])
        self._change_instructions()

    def rotate(self, angle):
        self.freeAngle += angle
        self.refresh()

    def setCenterPointCoordinates(self, x, y):
        self.set_center_point((x, y))

    def translate(self, x, y):
        self.set_center_point((x, y))
        self.refresh()

    def load_image_file(self, fileName):
        self._rectangle_instruction.source = fileName

    def bounce_x(self):
        if self.speedPixelsPerFrameXYPair is not None:
            self.speedPixelsPerFrameXYPair = self.speedPixelsPerFrameXYPair[0] * -1.0, self.speedPixelsPerFrameXYPair[1]

    def bounce_y(self):
        if self.speedPixelsPerFrameXYPair is not None:
            self.speedPixelsPerFrameXYPair = self.speedPixelsPerFrameXYPair[0], self.speedPixelsPerFrameXYPair[1] * -1.0

    def setScale(self, multiplier):
        self._scale_instruction.x = float(multiplier)
        self._scale_instruction.y = float(multiplier)
        self._change_instructions()

    def setScaleXY(self, xMultiplier, yMultiplier):
        self._scale_instruction.x = float(xMultiplier)
        self._scale_instruction.y = float(yMultiplier)
        self._change_instructions()

    # region color
    def setColor(self, color):
        try:
            self._color_instruction.b = color.b
            self._color_instruction.g = color.g
            self._color_instruction.r = color.r
            self._color_instruction.a = color.a
        except:
            try:
                self._color_instruction.b = color[0]
                self._color_instruction.g = color[1]
                self._color_instruction.r = color[2]
                self._color_instruction.a = color[3]
            except:
                pass

    def getColorR(self):
        return self._color_instruction.r

    def get_id(self):
        return self.name

    def getColorG(self):
        return self._color_instruction.g

    def getColorB(self):
        return self._color_instruction.b

    def getColorA(self):
        return self._color_instruction.a

    def get_grid(self):
        return self

    def get_number(self):
        return 1

    def update_virtual_pos(self):
        pass
    # endregion color
# end class SpriteWidget