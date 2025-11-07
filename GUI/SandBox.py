from kivy.graphics import Color, Rectangle, Line, InstructionGroup

class SandBox(InstructionGroup):
    def __init__(self, size, window_size):
        super(SandBox, self).__init__()
        self.padding = 0
        self.outline_width = 2

        self.window_w, self.window_h = window_size
        win_w, win_h = window_size

        w, h = size
        w = max(1, int(w + 2 * self.padding))
        h = max(1, int(h + 2 * self.padding))

        center = (win_w / 2.0, win_h / 2.0)
        cx, cy = center
        self.sandbox_center = (cx, cy)
        self.sandbox_size = size

        x = int(cx - w / 2)
        y = int(cy - h / 2)

        # clamp into window (so we don't draw negative sizes)
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

        print("border rectangle", self.border_line.rectangle)
        # self.add(Color(1, 0, 0, 1))  # full red
        # self.add(Line(rectangle=(x, y, w, h), width=self.outline_width))

    def set_box(self, x, y, w, h):
        """
        Set sandbox box directly: (x, y, w, h) in window coords.
        Updates border + all 4 overlay rectangles.
        """
        # padding
        x -= self.padding
        y -= self.padding
        w += 2 * self.padding
        h += 2 * self.padding

        # clamp into window
        x = max(0, min(x, self.window_w - 1))
        y = max(0, min(y, self.window_h - 1))
        w = max(1, min(w, self.window_w - x))
        h = max(1, min(h, self.window_h - y))

        # --- border line in middle (unfilled box) ---
        self.border_line.rectangle = (x, y, w, h)

        # --- transparent areas outside the box ---

        # Left of box (full height)
        self.left_rectangle.pos = (0, 0)
        self.left_rectangle.size = (max(0, x), self.window_h)

        # Right of box (full height)
        right_x = x + w
        self.right_rectangle.pos = (right_x, 0)
        self.right_rectangle.size = (max(0, self.window_w - right_x), self.window_h)

        # Bottom strip under box
        self.bottom_rectangle.pos = (x, 0)
        self.bottom_rectangle.size = (w, max(0, y))

        # Top strip above box
        top_y = y + h
        self.top_rectangle.pos = (x, top_y)
        self.top_rectangle.size = (w, max(0, self.window_h - top_y))

    def set_box_centered(self, size):
        """Convenience: center box in window with given (w, h)."""
        w, h = size
        cx, cy = self.window_w / 2.0, self.window_h / 2.0
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        self.set_box(x, y, w, h)