from kivy.graphics import Color, Rectangle, Line, InstructionGroup

class SandBox(InstructionGroup):
    def __init__(self, size, center, window_size, cm_to_px=1.0):
        super(SandBox, self).__init__()
        self.padding = 0
        self.outline_width = 2

        self.sandbox_size = size
        cx, cy = center[0], center[1]
        self.sandbox_center = (cx, cy)

        self.window_w, self.window_h = window_size

        win_w, win_h = window_size

        w, h = size
        w = max(1, int(w + 2 * self.padding))
        h = max(1, int(h + 2 * self.padding))
        center = center
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        x = max(0, min(x, win_w - w))
        y = max(0, min(y, win_h - h))

        self.left_rectangle = Rectangle()
        self.right_rectangle = Rectangle()
        self.bottom_rectangle = Rectangle()
        self.top_rectangle = Rectangle()

        self.left_rectangle.pos = (0, y - w)
        self.left_rectangle.size = (x, h + 2 * w)
        self.right_rectangle.pos = (x + w, y - w)
        self.right_rectangle.size = (max(0, win_w - (x + w)), h + 2 * w)
        self.bottom_rectangle.pos = (x, 0)
        self.bottom_rectangle.size = (w, y)
        self.top_rectangle.pos = (x, y + h)
        self.top_rectangle.size = (w, max(0, win_h - (y + h)))

        self.add(Color(1, 0, 0, 0.35))  # transparent red

        self.add(self.left_rectangle)
        self.add(self.right_rectangle)
        self.add(self.bottom_rectangle)
        self.add(self.top_rectangle)

        self.border_color = Color(1, 0, 0, 1) # full red
        self.border_line = Line(rectangle=(x, y, w, h), width=self.outline_width)
        self.add(self.border_line)

