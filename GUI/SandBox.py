from kivy.graphics import Color, Rectangle, Line, InstructionGroup, Scale, PushMatrix, Translate, Rotate, PopMatrix


class SandBox(InstructionGroup):
    def __init__(self, size, center, window_size, cm_to_px=1.0):
        super(SandBox, self).__init__()
        self.padding = 0
        self.outline_width = 2

        self.base_scale_factor = 1.0
        self.true_scale = 1.0

        self.sandbox_size = size
        self.sandbox_center = center

        self.window_w, self.window_h = window_size
        self.size_w, self.size_h = size

        self.size_w = max(1, int(self.size_w + 2 * self.padding))
        self.size_h = max(1, int(self.size_h + 2 * self.padding))

        x = int(self.sandbox_center[0] - self.size_w / 2)
        y = int(self.sandbox_center[1] - self.size_h / 2)

        self.x = max(0, min(x, self.window_w - self.size_w))
        self.y = max(0, min(y, self.window_h - self.size_h))

        self.left_rectangle = Rectangle()
        self.right_rectangle = Rectangle()
        self.bottom_rectangle = Rectangle()
        self.top_rectangle = Rectangle()

        self.left_rectangle.pos = (0, self.y - self.size_w)
        self.left_rectangle.size = (self.x, self.size_h + 2 * self.size_w)
        self.right_rectangle.pos = (self.x + self.size_w, self.y - self.size_w)
        self.right_rectangle.size = (max(0, self.window_w - (self.x + self.size_w)), self.size_h + 2 * self.size_w)
        self.bottom_rectangle.pos = (self.x, 0)
        self.bottom_rectangle.size = (self.size_w, self.y)
        self.top_rectangle.pos = (self.x, self.y + self.size_h)
        self.top_rectangle.size = (self.size_w, max(0, self.window_h - (self.y + self.size_h)))
        self.left_rectangle_border_pos = (self.left_rectangle.pos[0] + self.left_rectangle.size[0], self.left_rectangle.pos[1])
        self.right_rectangle_border_pos = (self.right_rectangle.pos[0], self.right_rectangle.pos[1])
        self.bottom_rectangle_border_pos = (self.bottom_rectangle.pos[0] ,self.bottom_rectangle.pos[1] + self.bottom_rectangle.size[1])
        self.top_rectangle_border_pos = (self.top_rectangle.pos[0], self.top_rectangle.pos[1])


        self.border_color = Color(1, 0, 0, 1) # full red
        self.border_line = Line(rectangle=(self.x, self.y, self.size_w, self.size_h), width=self.outline_width)

        self.scale_factor = self.base_scale_factor
        self.zoom_scale = Scale(x=self.scale_factor, y=self.scale_factor, origin=(0, 0))

        self.trans = Translate(0, 0)

        #Instruction Matrix
        self.add(PushMatrix())

        self.add(self.zoom_scale)
        self.add(self.trans)

        self.add(Color(1, 0, 0, 0.35))  # transparent red

        self.add(self.left_rectangle)
        self.add(self.right_rectangle)
        self.add(self.bottom_rectangle)
        self.add(self.top_rectangle)

        self.add(self.border_line)

        self.add(PopMatrix())

        self.zoom_matrix = self.zoom_scale.matrix.tolist()
        self.trans_matrix = self.trans.matrix.tolist()
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix)

    def zoom_default(self, factor, origin):
        self.zoom_scale.origin = origin
        self.scale_factor = factor
        self.zoom_scale.x = factor
        self.zoom_scale.y = factor
        self.base_scale_factor = factor
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix)

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
        self.transform_matrix = self.zoom_scale.matrix.multiply(self.trans.matrix)

        self.reset_rectangle_size()


    def zoom_reset(self):
        self.scale_factor = 1.0
        self.base_scale_factor = 1.0
        self.zoom_scale.x = 1.0
        self.zoom_scale.y = 1.0
        self.reset_rectangle_size()

    def reset_rectangle_size(self):
        self.max_scale_size = (8, 8)
        # self.left_rectangle.pos = self.left_rectangle.pos
        self.left_rectangle.size = self.max_scale_size[0] * self.window_h, self.max_scale_size[1] * self.window_w
        self.left_rectangle.pos = (self.left_rectangle_border_pos[0] - self.left_rectangle.size[0],
                                   self.left_rectangle_border_pos[1] - self.left_rectangle.size[1]/2 + self.size_h/2)
        # self.right_rectangle.pos = (x + w, y - w)
        self.right_rectangle.size = self.max_scale_size[0] * self.window_h,  self.max_scale_size[1] * self.window_w
        self.right_rectangle.pos = self.right_rectangle_border_pos[0], self.right_rectangle_border_pos[1] - self.right_rectangle.size[1]/2 + self.size_h/2
        # self.bottom_rectangle.pos = (x, 0)
        self.bottom_rectangle.size = self.bottom_rectangle.size[0], self.max_scale_size[1] * self.window_w
        self.bottom_rectangle.pos = self.bottom_rectangle.pos[0], self.bottom_rectangle_border_pos[1] - self.bottom_rectangle.size[1]
        # self.top_rectangle.pos = (x, y + h)
        self.top_rectangle.size = self.top_rectangle.size[0], self.max_scale_size[1] * self.window_w

