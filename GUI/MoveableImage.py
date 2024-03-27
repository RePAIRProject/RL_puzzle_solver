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


class MovableImage(DragBehavior, Image):
    def __init__(self, source, label, score, image_number, has_score):
        super(MovableImage, self).__init__()
        global score_label
        self.limit_image = self
        self.source = source
        self.drag_timeout = 10000000
        self.drag_distance = 0

        # self.size_hint = (.13, .13)
        # self.keep_ratio = True
        # self.allow_stretch = True

        self.image_number = image_number
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        # self.rotate = Rotate(angle)
        self.label = label
        self.score = score
        # self.canvas.before.add(PushMatrix())
        # self.canvas.before.add(self.rotate)
        # self.canvas.after.add(PopMatrix())

        self.bind()
        self.scatter = Scatter()

        self.grid = GridLayout()
        self.grid.cols = 1
        self.grid.rows = 2

        self.grid.add_widget(self)

        if has_score:
            score_label = Label()
            score_label.text = str(self.score)
            self.grid.add_widget(score_label)

        # self.movable_layout = MovableLayout()
        # self.movable_layout.add_widget(self)

        # self.scatter.add_widget(self)
        # self.bind(self.update_canvas)

    # label = Label()

    # def set_label(self, label):
    #     self.label = label

    def get_grid(self):
        return self.grid

    def remove_score(self):
        global score_label
        self.grid.remove(score_label)

    def get_scatter(self):
        return self.scatter

    def update_canvas(self, *args):
        self.rotate.origin = self.center

    # def on_touch_down(self, touch):
    #     if self.collide_point(*touch.pos):
    #         click_label.text = str(self.image_number + 1)

    def add_score(self, *args, **kwargs):
        self.grid.add_widget(self.score_label)

    def on_pos(self, *args):
        # print("here")
        self.drag_rectangle = [self.x, self.y, self.width, self.height]
        self.label.text = str(self.image_number + 1)
        print("here" + str(self.image_number))
        # click_label.text = str(self.image_number + 1)
        # self.label = self.image_number + 1

    # def on_touch_up(self, touch):
    #     self.label.text = str(self.image_number + 1)
    #     print("here" + str(self.image_number))

    # def on_touch_up(self, *args):
    #     print('Released split1_bottom bar')
    #     print('Y value = %d' % self.y)
    #     self.drag_rectangle = [self.x, self.y, self.width, self.height]
    #     self.label.text = str(self.image_number + 1)

    # def on_touch_move(self, touch):
    #     if touch.grab_current is self:
    #         self.inside_limit_image(touch)  # keep this MoveableImage within the limit_image
    #     return super(MovableImage, self).on_touch_move(touch)

    def on_size(self, *args):
        self.drag_rectangle = [self.x, self.y, self.width, self.height]

    def inside_limit_image(self, touch):
        if self.limit_image is None:
            return

        # calculate limits of actual picture inside this MoveableImage
        m_image_min_x = self.x + (self.width - self.norm_image_size[0]) / 2.
        m_image_min_y = self.y + (self.height - self.norm_image_size[1]) / 2.
        m_image_max_x = m_image_min_x + self.norm_image_size[0]
        m_image_max_y = m_image_min_y + self.norm_image_size[1]

        # calculate where limits of picture in the MoveableImage would be if move is allowed
        new_min = [m_image_min_x + touch.dx, m_image_min_y + touch.dy]
        new_max = [new_min[0] + self.norm_image_size[0], new_min[1] + self.norm_image_size[1]]

        # calculate limits of picture in the limit_image
        image_min_x = self.limit_image.x + (self.limit_image.width - self.limit_image.norm_image_size[0]) / 2.
        image_min_y = self.limit_image.y + (self.limit_image.height - self.limit_image.norm_image_size[1]) / 2.
        image_max_x = image_min_x + self.limit_image.norm_image_size[0]
        image_max_y = image_min_y + self.limit_image.norm_image_size[1]

        # adjust touch, if necessary, to keep MoveableImage within limit_image
        if new_min[0] < image_min_x:
            touch.dx = image_min_x - m_image_min_x
        if new_min[1] < image_min_y:
            touch.dy = image_min_y - m_image_min_y
        if new_max[0] > image_max_x:
            touch.dx = image_max_x - m_image_max_x
        if new_max[1] > image_max_y:
            touch.dy = image_max_y - m_image_max_y
        return
