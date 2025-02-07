from kivy.clock import Clock
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Color, Translate, Rotate
from kivy.uix.widget import Widget
from kivy.resources import resource_find
from kivy.graphics import RenderContext, Callback, UpdateNormalMatrix, Mesh
from objloader import ObjFile
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import glEnable, glDisable, GL_DEPTH_TEST


class Widget3D(Widget):
    def __init__(self, object_path, shader_path, **kwargs):
        self.canvas = RenderContext()
        self.canvas.shader.source = resource_find(shader_path)
        self.scene = ObjFile(resource_find(object_path))
        super(Widget3D, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['projection_mat'] = proj
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        # Clock.schedule_interval(self.update_glsl, 1 / 60.)

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['projection_mat'] = proj
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.rot.angle += delta * 100

    def create_mesh_chunks(vertices, indices, chunk_size=65535):
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i + chunk_size]
            chunk_vertices = vertices  # Adjust this if necessary
            yield Mesh(vertices=chunk_vertices, indices=chunk_indices, mode='triangles')

    def setup_scene(self):
        Color(1, 1, 1, 1)
        PushMatrix()
        Translate(0, 0, -3)
        self.rot = Rotate(1, 0, 1, 0)

        if not self.scene.objects:
            raise ValueError('No objects found in the loaded .obj file.')

        try:
            m = list(self.scene.objects.values())[0]
            UpdateNormalMatrix()
            # self.create_mesh_chunks(m.vertices, m.indices)
            self.mesh = Mesh(
                vertices=m.vertices,
                indices=m.indices,
                fmt=m.vertex_format,
                index_type='uint32',
                mode='triangles',
            )
        except IndexError:
            raise IndexError(
                "Failed to access objects in the scene. Ensure the .obj file is valid and contains objects.")
        PopMatrix()

# from kivy3 import Scene, PerspectiveCamera, Material, Mesh
# from kivy3.extras.geometries import BoxGeometry
# from Renderer import Renderer
#
# class Widget3D(Renderer):
#     def __init__(self, **kwargs):
#         super(Widget3D,  self).__init__(**kwargs)
#         scene = Scene()
#
#         cube_geo = BoxGeometry(1, 1, 1)
#         cube_mat = Material()
#         self.cube = Mesh(
#             geometry=cube_geo,
#             material=cube_mat
#         )  # default pos == (0, 0, 0)
#         self.cube.pos.z = -5
#
#         self.camera = PerspectiveCamera(
#             fov=15,  # distance from the screen
#             aspect=1,  # "screen" ratio
#             near=1,  # nearest rendered point
#             far=10  # farthest rendered point
#         )
#
#         # start rendering the scene and camera
#         scene.add(self.cube)
#         self.render(scene, self.camera)
#
#         # self.bind(size=self._adjust_aspect)
