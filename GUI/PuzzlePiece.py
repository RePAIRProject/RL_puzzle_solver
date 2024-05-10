class PuzzlePiece:
    def __init__(self, name, pos=None, theta=None):
        self.name = name
        self.pos = pos
        self.theta = theta

    def set_pos(self, pos):
        self.pos = pos

    def set_theta(self, theta):
        self.theta = theta

    def get_pos(self):
        return self.pos

    def get_theta(self):
        return self.theta
