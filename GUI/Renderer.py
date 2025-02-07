import os
import kivy3


from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.fbo import Fbo
from kivy.core.window import Window
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST
from kivy.graphics.transformation import Matrix
from kivy.graphics import (
    Callback, PushMatrix, PopMatrix,
    Rectangle, Canvas, UpdateNormalMatrix
)

from kivy3.light import Light

kivy3_path = os.path.abspath(os.path.dirname(kivy3.__file__))

class Renderer(Widget):

    def __init__(self, **kw):
        self.shader_file = kw.pop("shader_file", None)
        self.canvas = Canvas()
        super(Renderer, self).__init__(**kw)

        with self.canvas:
            self._viewport = Rectangle(size=self.size, pos=self.pos)
            self.fbo = Fbo(size=self.size,
                           with_depthbuffer=True, compute_normal_mat=True,
                           clear_color=(0., 0., 0., 0.))
        self._config_fbo()
        self.texture = self.fbo.texture
        self.camera = None
        self.scene = None
        self.main_light = Light(renderer=self)

    def _config_fbo(self):
        # set shader file here
        os_path = os.getcwd() + "/GUI/3D"
        self.fbo.shader.source = os.path.join(os_path, "default.glsl")
        with self.fbo:
            Callback(self._setup_gl_context)
            PushMatrix()
            # instructions set for all instructions
            self._instructions = InstructionGroup()
            PopMatrix()
            Callback(self._reset_gl_context)

    def _setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)
        self.fbo.clear_buffer()

    def _reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def render(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.camera.bind_to(self)
        self._instructions.add(scene.as_instructions())
        Clock.schedule_once(self._update_matrices, -1)

    def on_size(self, instance, value):
        self.fbo.size = value
        self._viewport.texture = self.fbo.texture
        self._viewport.size = value
        self._viewport.pos = self.pos
        self._update_matrices()

    def on_pos(self, instance, value):
        self._viewport.pos = self.pos
        self._update_matrices()

    def on_texture(self, instance, value):
        self._viewport.texture = value

    def _update_matrices(self, dt=None):
        if self.camera:
            self.fbo['projection_mat'] = self.camera.projection_matrix
            self.fbo['modelview_mat'] = self.camera.modelview_matrix
            self.fbo['model_mat'] = self.camera.model_matrix
            self.fbo['camera_pos'] = [float(p) for p in self.camera.position]
            self.fbo['view_mat'] = Matrix().rotate(
                Window.rotation, 0.0, 0.0, 1.0)
        else:
            raise RendererError("Camera is not defined for renderer")

    def set_clear_color(self, color):
        self.fbo.clear_color = color