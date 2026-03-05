"""QOpenGLWidget - 3D star map with VBOs, shaders, camera, mouse interaction.
Ported from MainForm.cs GL rendering logic."""
import os
import sys
import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QFont, QColor, QFontDatabase, QFontMetrics, QBrush

from OpenGL.GL import *
import ctypes

from taco.core.solar_system_manager import SolarSystemManager
from taco.core.solar_system import color_to_rgba32, DEFAULT_DRAW_COLOR
from taco.rendering.shader import Shader
from taco.rendering.texture_loader import load_texture
from taco.rendering.mouse_ray import MouseRay
from taco.rendering.font_atlas import FontAtlas
from taco.rendering.text_renderer import TextRenderer


def _resource_path(relative: str) -> str:
    if getattr(sys, 'frozen', False):
        base = os.path.join(sys._MEIPASS, "taco", "resources")
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources")
    return os.path.normpath(os.path.join(base, relative))


class GLMapWidget(QOpenGLWidget):
    system_clicked = pyqtSignal(int)       # system id
    system_hovered = pyqtSignal(int, str)  # system id, name
    system_right_clicked = pyqtSignal(int, object)  # system id, QPoint

    def __init__(self, manager: SolarSystemManager, parent=None):
        super().__init__(parent)
        self.manager = manager

        # Camera
        self._camera_distance = 2000.0
        self._look_at = np.array([-1416.0, 3702.0, 0.0], dtype=np.float32)
        self._eye = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._vec_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._point_size = 1.0
        self._crosshair_size = 26.0

        # Matrices
        self._projection = np.eye(4, dtype=np.float32)
        self._modelview = np.eye(4, dtype=np.float32)

        # Mouse
        self._dragging = False
        self._start_x = 0
        self._start_y = 0
        self._current_highlight = -1
        self._clicked_system = -1

        # Zoom animation
        self._zooming = False
        self._zoom_tick = 0
        self._max_zoom_tick = 100
        self._zoom_start = np.zeros(3, dtype=np.float32)
        self._zoom_end = np.zeros(3, dtype=np.float32)
        self._zoom_to_system_id = -1

        # GL objects
        self._gl_loaded = False
        self._shaders_loaded = False
        self._shader = None
        self._shader_conn = None
        self._shader_crosshair = None

        # VBO/VAO IDs
        self._system_vbo = 0
        self._system_vao = 0
        self._color_vbo = 0
        self._conn_vbo = 0
        self._conn_vao = 0
        self._conn_color_vbo = 0
        self._crosshair_vbo = 0
        self._crosshair_vao = 0

        # Textures
        self._tex_system = 0
        self._tex_green_ch = 0
        self._tex_red_ch = 0
        self._tex_yellow_ch = 0
        self._tex_red_green_ch = 0
        self._tex_red_yellow_ch = 0
        self._tex_yellow_green_ch = 0

        # Render state
        self._is_highlighting = False
        self._has_rendered = False

        # Font
        self._font = None
        self._font_loaded = False
        self._map_text_size = 8
        # Cached fonts/metrics — rebuilt when _map_text_size changes
        self._star_font: QFont | None = None
        self._star_fm: QFontMetrics | None = None
        self._char_font: QFont | None = None
        self._char_fm: QFontMetrics | None = None
        self._alert_font: QFont | None = None
        self._alert_fm: QFontMetrics | None = None
        self._popup_font: QFont | None = None
        self._popup_fm: QFontMetrics | None = None
        self._region_font: QFont | None = None
        self._region_fm: QFontMetrics | None = None
        self._region_font_pt: int = -1

        # GPU text rendering
        self._shader_text: Shader | None = None
        self._text_renderer: TextRenderer | None = None
        self._atlas_scale: float = 2.0
        self._atlas_star: FontAtlas | None = None
        self._atlas_star_bold: FontAtlas | None = None
        self._atlas_char: FontAtlas | None = None
        self._atlas_alert: FontAtlas | None = None
        self._atlas_popup: FontAtlas | None = None
        self._atlas_region: FontAtlas | None = None
        self._atlas_region_pt: int = -1
        self._gpu_text_ready = False

        self._persistent_labels = False
        self._show_alert_age = True
        self._display_char_names = True
        self._show_char_locations = True

        # Sticky highlights for persistent labels
        self._sticky_highlight_systems: set[int] = set()

        # Landmark systems (always-visible labels regardless of zoom)
        self._landmark_systems: set[int] = set()

        # Map mode for region label visibility
        self._map_mode: str = "3d"

        # Character locations: char_name -> system_id
        self._char_locations: dict[str, int] = {}

        # Alert label screen rects for hover hit-testing: sys_id -> QRect
        self._alert_label_rects: dict[int, QRect] = {}
        # Crosshair icon screen rects for hover hit-testing: sys_id -> QRect
        self._crosshair_icon_rects: dict[int, QRect] = {}

        # Cached flat_colors for shader uniforms (rebuilt only when uniforms change)
        self._cached_flat_colors: list[float] = []

        # Cached jump distances for alert labels: (from_sys, to_sys) -> jumps
        self._jump_distance_cache: dict[tuple[int, int], int] = {}

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)  # ~30fps
        self._anim_timer.timeout.connect(self._on_anim_tick)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --- Properties ---

    @property
    def camera_distance(self) -> float:
        return self._camera_distance

    @camera_distance.setter
    def camera_distance(self, value: float):
        self._camera_distance = max(10.0, min(15000.0, value))

    @property
    def look_at(self) -> np.ndarray:
        return self._look_at

    @look_at.setter
    def look_at(self, value: np.ndarray):
        self._look_at = value

    @property
    def map_text_size(self) -> int:
        return self._map_text_size

    @map_text_size.setter
    def map_text_size(self, value: int):
        self._map_text_size = max(4, min(24, value))
        self._load_font()
        self._rebuild_cached_fonts()
        if self._gpu_text_ready:
            self.makeCurrent()
            self._rebuild_gpu_atlases()
            self.doneCurrent()

    @property
    def persistent_labels(self) -> bool:
        return self._persistent_labels

    @persistent_labels.setter
    def persistent_labels(self, value: bool):
        self._persistent_labels = value
        self.update()

    @property
    def show_alert_age(self) -> bool:
        return self._show_alert_age

    @show_alert_age.setter
    def show_alert_age(self, value: bool):
        self._show_alert_age = value

    @property
    def display_char_names(self) -> bool:
        return self._display_char_names

    @display_char_names.setter
    def display_char_names(self, value: bool):
        self._display_char_names = value

    @property
    def show_char_locations(self) -> bool:
        return self._show_char_locations

    @show_char_locations.setter
    def show_char_locations(self, value: bool):
        self._show_char_locations = value

    @property
    def sticky_highlight_systems(self) -> set[int]:
        return self._sticky_highlight_systems

    @sticky_highlight_systems.setter
    def sticky_highlight_systems(self, value: set[int]):
        self._sticky_highlight_systems = value
        self.update()

    @property
    def landmark_systems(self) -> set[int]:
        return self._landmark_systems

    @landmark_systems.setter
    def landmark_systems(self, value: set[int]):
        self._landmark_systems = value
        self.update()

    @property
    def char_locations(self) -> dict[str, int]:
        return self._char_locations

    @char_locations.setter
    def char_locations(self, value: dict[str, int]):
        self._char_locations = value
        self._jump_distance_cache.clear()
        self.update()

    def set_map_mode(self, mode: str):
        self._map_mode = mode
        self.update()

    def start_animation(self):
        self._anim_timer.start()

    def stop_animation(self):
        self._anim_timer.stop()

    def _on_anim_tick(self):
        has_animations = self.manager.incoming_tick()
        if has_animations:
            self.manager.build_uniforms()
            self._cached_flat_colors.clear()
            self._jump_distance_cache.clear()
        self.manager.process_pathfinding_queue()
        self.manager.remove_expired_alerts()
        self.update()

    # --- OpenGL ---

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        try:
            glEnable(GL_POINT_SPRITE)
        except Exception:
            pass  # Not needed in OpenGL 3.3+ Core profile

        self._load_shaders()
        self._load_textures()
        self._init_vbos()
        self._load_font()
        self._init_gpu_text()

        self._gl_loaded = True

    def _load_font(self):
        font_path = _resource_path("fonts/Taco.ttf")
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                self._font = QFont(families[0], self._map_text_size)
            else:
                self._font = QFont("Monospace", self._map_text_size)
        else:
            self._font = QFont("Monospace", self._map_text_size)
        self._font_loaded = True
        self._rebuild_cached_fonts()

    def _make_verdana_font(self, pt: int) -> QFont:
        f = QFont("Verdana", pt)
        f.setFamilies(["Verdana", "DejaVu Sans", "Liberation Sans"])
        f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
        return f

    def _rebuild_cached_fonts(self):
        sz = self._map_text_size
        self._star_font = self._make_verdana_font(sz)
        self._star_fm = QFontMetrics(self._star_font)
        self._char_font = self._make_verdana_font(max(sz + 1, 8))
        self._char_fm = QFontMetrics(self._char_font)
        self._alert_font = self._make_verdana_font(max(sz + 1, 7))
        self._alert_fm = QFontMetrics(self._alert_font)
        self._popup_font = self._make_verdana_font(max(sz, 7))
        self._popup_fm = QFontMetrics(self._popup_font)
        # Region font is zoom-dependent; invalidate so it rebuilds next frame
        self._region_font_pt = -1

    def _init_gpu_text(self):
        """Initialize GPU text rendering resources (shader, renderer, atlases)."""
        shader_dir = _resource_path("shaders")

        def read_shader(name):
            with open(os.path.join(shader_dir, name), 'r', encoding='utf-8-sig') as f:
                return f.read()

        self._shader_text = Shader(read_shader("text.vert"), read_shader("text.frag"))
        self._text_renderer = TextRenderer(self._shader_text)
        self._text_renderer.init_gl()
        self._rebuild_gpu_atlases()
        self._gpu_text_ready = True

    def _rebuild_gpu_atlases(self):
        """Rebuild font atlases for current text size. Requires GL context."""
        sz = self._map_text_size
        scale = max(self.devicePixelRatioF(), 2.0)
        self._atlas_scale = scale
        # Dispose old atlases
        for atlas in (self._atlas_star, self._atlas_star_bold, self._atlas_char,
                      self._atlas_alert, self._atlas_popup, self._atlas_region):
            if atlas:
                atlas.dispose()

        self._atlas_star = FontAtlas(self._make_verdana_font(sz), scale=scale)
        self._atlas_star.upload()
        self._atlas_star_bold = FontAtlas(self._make_verdana_font(sz), bold=True, scale=scale)
        self._atlas_star_bold.upload()
        self._atlas_char = FontAtlas(self._make_verdana_font(max(sz + 1, 8)), scale=scale)
        self._atlas_char.upload()
        self._atlas_alert = FontAtlas(self._make_verdana_font(max(sz + 1, 7)), scale=scale)
        self._atlas_alert.upload()
        self._atlas_popup = FontAtlas(self._make_verdana_font(max(sz, 7)), scale=scale)
        self._atlas_popup.upload()
        # Region atlas built on-demand at zoom-dependent size
        self._atlas_region = None
        self._atlas_region_pt = -1

    def _get_region_font(self, pt: int) -> tuple[QFont, QFontMetrics]:
        if pt != self._region_font_pt:
            self._region_font = QFont("Verdana", pt)
            self._region_font.setFamilies(["Verdana", "DejaVu Sans", "Liberation Sans"])
            self._region_font.setBold(True)
            self._region_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
            self._region_fm = QFontMetrics(self._region_font)
            self._region_font_pt = pt
        return self._region_font, self._region_fm

    def _load_shaders(self):
        shader_dir = _resource_path("shaders")

        def read_shader(name):
            with open(os.path.join(shader_dir, name), 'r', encoding='utf-8-sig') as f:
                return f.read()

        self._shader = Shader(read_shader("shader.vert"), read_shader("shader.frag"))
        self._shader_conn = Shader(read_shader("connection.vert"), read_shader("connection.frag"))
        self._shader_crosshair = Shader(read_shader("crosshair.vert"), read_shader("crosshair.frag"))
        self._shaders_loaded = True

    def _load_textures(self):
        tex_dir = _resource_path("textures")
        self._tex_system = load_texture(os.path.join(tex_dir, "system.png"))
        self._tex_green_ch = load_texture(os.path.join(tex_dir, "green-crosshair.png"))
        self._tex_red_ch = load_texture(os.path.join(tex_dir, "red-crosshair.png"))
        self._tex_yellow_ch = load_texture(os.path.join(tex_dir, "yellow-crosshair.png"))
        self._tex_red_green_ch = load_texture(os.path.join(tex_dir, "redgreen-crosshair.png"))
        self._tex_red_yellow_ch = load_texture(os.path.join(tex_dir, "redyellow-crosshair.png"))
        self._tex_yellow_green_ch = load_texture(os.path.join(tex_dir, "yellowgreen-crosshair.png"))

    def _init_vbos(self):
        if self.manager.system_count == 0:
            return

        self.manager.init_vbo_data()

        # System points VBO
        self._system_vao = glGenVertexArrays(1)
        glBindVertexArray(self._system_vao)

        self._system_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._system_vbo)
        data = self.manager.system_vbo_content
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # System color VBO
        self._color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo)
        color_data = self.manager.system_color_vao_content
        glBufferData(GL_ARRAY_BUFFER, color_data.nbytes, color_data, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, None)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

        # Connection VBO
        if self.manager.connection_vertex_count > 0:
            self._conn_vao = glGenVertexArrays(1)
            glBindVertexArray(self._conn_vao)

            self._conn_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._conn_vbo)
            conn_data = self.manager.connection_vbo_content
            glBufferData(GL_ARRAY_BUFFER, conn_data.nbytes, conn_data, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            self._conn_color_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._conn_color_vbo)
            conn_color = self.manager.connection_color_vao_content
            glBufferData(GL_ARRAY_BUFFER, conn_color.nbytes, conn_color, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            glBindVertexArray(0)

        # Crosshair VBO (single-point, updated dynamically)
        self._crosshair_vao = glGenVertexArrays(1)
        glBindVertexArray(self._crosshair_vao)
        self._crosshair_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._crosshair_vbo)
        dummy = np.zeros((1, 3), dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, dummy.nbytes, dummy, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _update_vbos(self):
        if self._color_vbo == 0:
            return
        self.manager.refresh_vbo_data()

        # Re-upload system position VBO when coordinates changed (e.g. map mode switch)
        if self.manager.is_system_vbo_dirty and self._system_vbo != 0:
            glBindBuffer(GL_ARRAY_BUFFER, self._system_vbo)
            data = self.manager.system_vbo_content
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            self.manager.is_system_vbo_dirty = False

        if self.manager.is_color_vao_dirty:
            glBindBuffer(GL_ARRAY_BUFFER, self._color_vbo)
            color_data = self.manager.system_color_vao_content
            glBufferSubData(GL_ARRAY_BUFFER, 0, color_data.nbytes, color_data)
            self.manager.is_color_vao_dirty = False

        # Re-upload connection VBOs when positions changed (e.g. map mode switch)
        if self.manager.is_connection_vbo_data_dirty:
            self.manager._extract_connections()
            if self._conn_vbo != 0 and self.manager.connection_vertex_count > 0:
                glBindBuffer(GL_ARRAY_BUFFER, self._conn_vbo)
                conn_data = self.manager.connection_vbo_content
                glBufferSubData(GL_ARRAY_BUFFER, 0, conn_data.nbytes, conn_data)
                glBindBuffer(GL_ARRAY_BUFFER, self._conn_color_vbo)
                conn_color = self.manager.connection_color_vao_content
                glBufferSubData(GL_ARRAY_BUFFER, 0, conn_color.nbytes, conn_color)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def paintGL(self):
        if not self._gl_loaded or not self._shaders_loaded:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        w = self.width()
        h = max(self.height(), 1)

        # Build projection
        aspect = w / h
        self._projection = self._perspective(45.0, aspect, 1.0, 50000.0)

        # Build modelview (camera)
        self._eye[0] = self._look_at[0]
        self._eye[1] = self._look_at[1]
        self._eye[2] = self._camera_distance
        self._modelview = self._look_at_matrix(self._eye, self._look_at, self._vec_y)

        # Update VBOs if dirty (positions, colors, connections)
        self._update_vbos()

        # Update uniforms
        if not self.manager.are_uniforms_clean:
            self.manager.build_uniforms()

        # --- Draw region labels (behind connections) ---
        self._draw_region_labels()

        # --- Draw connections ---
        if self._shader_conn and self.manager.connection_vertex_count > 0:
            self._shader_conn.bind()
            self._shader_conn.set_uniform_mat4("projection", self._projection)
            self._shader_conn.set_uniform_mat4("modelView", self._modelview)

            glBindVertexArray(self._conn_vao)
            glDrawArrays(GL_LINES, 0, self.manager.connection_vertex_count)
            glBindVertexArray(0)
            Shader.unbind()

        # --- Draw systems ---
        if self._shader and self.manager.system_count > 0:
            self._point_size = max(1.0, 6000.0 / self._camera_distance)
            self._crosshair_size = min(52.0, max(26.0, 16000.0 / self._camera_distance))
            self._shader.bind()
            self._shader.set_uniform_mat4("projection", self._projection)
            self._shader.set_uniform_mat4("modelView", self._modelview)
            self._shader.set_uniform_1f("pointsize", self._point_size)

            # Set highlight uniforms
            self._shader.set_uniform_1iv("hlpoints", self.manager.uniform_systems)
            self._shader.set_uniform_1fv("hlsizes", self.manager.uniform_sizes)

            # Flatten colors to float array (cached, rebuilt when uniforms change)
            if not self._cached_flat_colors:
                flat_colors = []
                for c in self.manager.uniform_colors:
                    flat_colors.extend(c)
                self._cached_flat_colors = flat_colors
            self._shader.set_uniform_1fv("hlcolors", self._cached_flat_colors)

            # Bind system texture
            self._shader.bind_texture(self._tex_system, 0, "tex")

            glBindVertexArray(self._system_vao)
            glDrawArrays(GL_POINTS, 0, self.manager.system_count)
            glBindVertexArray(0)
            Shader.unbind()

        # --- Draw crosshairs ---
        self._draw_crosshairs()

        # --- Draw text labels (QPainter overlay) ---
        self._draw_labels()

        self._has_rendered = True

    def _draw_crosshairs(self):
        if not self._shader_crosshair:
            return

        def draw_crosshair_set(system_ids, texture_id):
            for sys_id in system_ids:
                if sys_id not in self.manager.solar_systems:
                    continue
                system = self.manager.solar_systems[sys_id]
                pos = system.xyz.reshape(1, 3)

                glBindBuffer(GL_ARRAY_BUFFER, self._crosshair_vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, pos.nbytes, pos)

                self._shader_crosshair.bind()
                self._shader_crosshair.set_uniform_mat4("projection", self._projection)
                self._shader_crosshair.set_uniform_mat4("modelView", self._modelview)
                self._shader_crosshair.set_uniform_1f("pointsize", self._crosshair_size)
                self._shader_crosshair.bind_texture(texture_id, 0, "tex")

                glBindVertexArray(self._crosshair_vao)
                glDrawArrays(GL_POINTS, 0, 1)
                glBindVertexArray(0)
                Shader.unbind()

        # Disable depth test so crosshairs draw on top of connections and systems.
        # Draw order (green → yellow → red) ensures red alerts appear topmost.
        glDisable(GL_DEPTH_TEST)

        # Green crosshairs (home system)
        draw_crosshair_set(self.manager.green_crosshair_ids, self._tex_green_ch)

        # Character location crosshairs (yellow) - drawn before red so alerts show on top
        if self._show_char_locations:
            char_system_ids = set()
            for sys_id in self._char_locations.values():
                if sys_id >= 0:
                    char_system_ids.add(sys_id)
            home = self.manager.home_system_id
            char_only = [sid for sid in char_system_ids if sid != home]
            draw_crosshair_set(char_only, self._tex_yellow_ch)

        # Red crosshairs (alerts) - drawn last so they appear on top of character icons
        red_ids = list(self.manager.red_crosshair_ids)
        draw_crosshair_set(red_ids, self._tex_red_ch)

        glEnable(GL_DEPTH_TEST)

    def _draw_region_labels(self):
        """Draw region name labels behind all 3D geometry."""
        if not self._gpu_text_ready or not self.manager.region_labels:
            return

        tr = self._text_renderer
        w, h = self.width(), self.height()
        tr.begin_frame(w, h)
        region_pt = max(20, min(40, int(40000 / self._camera_distance)))
        region_alpha_f = max(30, min(80, int(25000 / self._camera_distance))) / 255.0
        if region_pt != self._atlas_region_pt:
            if self._atlas_region:
                self._atlas_region.dispose()
            rfont = QFont("Verdana", region_pt)
            rfont.setFamilies(["Verdana", "DejaVu Sans", "Liberation Sans"])
            rfont.setBold(True)
            rfont.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality)
            self._atlas_region = FontAtlas(rfont, scale=getattr(self, '_atlas_scale', 2.0))
            self._atlas_region.upload()
            self._atlas_region_pt = region_pt
        atlas_r = self._atlas_region
        for name, rx, ry, rz in self.manager.region_labels:
            world_pos = np.array([rx, ry, rz], dtype=np.float32)
            screen_pos = self._project_to_screen(world_pos)
            if screen_pos is not None:
                sx, sy = screen_pos
                if -200 <= sx <= w + 200 and -200 <= sy <= h + 200:
                    tw, th = atlas_r.measure_text(name)
                    tr.add_text(sx - tw / 2, sy - th / 2, name, atlas_r,
                                1.0, 1.0, 1.0, region_alpha_f)
        tr.flush()

    def _draw_labels(self):
        """Draw 2D text labels on top of GL scene using GPU text renderer.

        Flushes in compositing layers (back-to-front) so that foreground
        elements properly occlude background ones with alpha blending.
        """
        if not self._gpu_text_ready or not self._has_rendered:
            return

        tr = self._text_renderer
        w, h = self.width(), self.height()

        # ── Layer 1: System name labels + character name boxes ──
        tr.begin_frame(w, h)

        systems_to_label = set()
        if self._current_highlight >= 0:
            systems_to_label.add(self._current_highlight)
        for sys_id in self._landmark_systems:
            systems_to_label.add(sys_id)
        if self._camera_distance < 2000:
            for sys_id in list(self.manager.red_crosshair_ids):
                systems_to_label.add(sys_id)
            for sys_id in list(self.manager.green_crosshair_ids):
                systems_to_label.add(sys_id)
            for sys_id in self._sticky_highlight_systems:
                systems_to_label.add(sys_id)
            for sys_id in self._char_locations.values():
                if sys_id >= 0:
                    systems_to_label.add(sys_id)
            if self._persistent_labels:
                for sys_id in self.manager.solar_systems:
                    systems_to_label.add(sys_id)

        # Collect connected systems for hovered star
        connected_ids: set[int] = set()
        if self._current_highlight >= 0 and self._current_highlight in self.manager.solar_systems:
            hl_system = self.manager.solar_systems[self._current_highlight]
            for conn in hl_system.connected_to:
                connected_ids.add(conn.to_system_id)
                systems_to_label.add(conn.to_system_id)

        # Systems with crosshairs — labels positioned to the left like character labels
        crosshair_ids = set(self.manager.green_crosshair_ids)
        crosshair_ids.update(self.manager.red_crosshair_ids)
        if self._show_char_locations:
            for sid in self._char_locations.values():
                if sid >= 0:
                    crosshair_ids.add(sid)

        atlas_s = self._atlas_star
        atlas_sb = self._atlas_star_bold
        region_names = self.manager._region_names
        red_ids_set = self.manager.red_crosshair_ids
        for sys_id in systems_to_label:
            if sys_id not in self.manager.solar_systems:
                continue
            # Hide system name label while alert is active (alert label shows it)
            if sys_id in red_ids_set and sys_id in crosshair_ids:
                continue
            system = self.manager.solar_systems[sys_id]
            screen_pos = self._project_to_screen(system.xyz)
            if screen_pos is not None:
                sx, sy = screen_pos
                if 0 <= sx <= w and 0 <= sy <= h:
                    label = system.name
                    if sys_id == self._clicked_system:
                        rname = region_names.get(system.region_id)
                        if rname:
                            label = f"{label} ({rname})"
                    use_bold = sys_id in connected_ids or sys_id == self._current_highlight
                    atlas = atlas_sb if use_bold else atlas_s
                    if sys_id in crosshair_ids:
                        # Position to the right of crosshair icon, vertically centered
                        # with opaque background box like character labels
                        tw, _ = atlas.measure_text(label)
                        ch_pad_x, ch_pad_y = 6, 4
                        bx = int(sx) + 14
                        by = int(sy) - (atlas.line_height + ch_pad_y * 2) // 2
                        tr.add_rect(bx, by, tw + ch_pad_x * 2,
                                    atlas.line_height + ch_pad_y * 2,
                                    0.0, 0.0, 0.0, 0.75,
                                    0.75, 0.75, 0.75, 0.85)
                        tr.add_text(bx + ch_pad_x, by + ch_pad_y,
                                    label, atlas,
                                    0.72, 0.72, 0.72, 0.82)
                    else:
                        tr.add_text(sx + 8, sy - 4 - atlas.ascent + atlas.line_height,
                                    label, atlas,
                                    0.72, 0.72, 0.72, 0.72)

        if self._display_char_names and self._show_char_locations:
            atlas_c = self._atlas_char
            pad_x, pad_y = 6, 4
            chars_by_system: dict[int, list[str]] = {}
            for char_name, sys_id in self._char_locations.items():
                if sys_id < 0 or sys_id not in self.manager.solar_systems:
                    continue
                chars_by_system.setdefault(sys_id, []).append(char_name)
            for sys_id, char_names in chars_by_system.items():
                system = self.manager.solar_systems[sys_id]
                screen_pos = self._project_to_screen(system.xyz)
                if screen_pos is not None:
                    sx, sy = screen_pos
                    if 0 <= sx <= w and 0 <= sy <= h:
                        box_h = atlas_c.line_height + pad_y * 2
                        total_h = box_h * len(char_names) + (len(char_names) - 1) * 2
                        start_y = int(sy) - total_h // 2
                        for i, char_name in enumerate(char_names):
                            tw, _ = atlas_c.measure_text(char_name)
                            box_x = int(sx) - 14 - tw - pad_x * 2
                            box_y = start_y + i * (box_h + 2)
                            tr.add_rect(box_x, box_y, tw + pad_x * 2, box_h,
                                        0.0, 0.0, 0.0, 0.75,
                                        0.75, 0.75, 0.75, 0.85)
                            tr.add_text(box_x + pad_x, box_y + pad_y, char_name, atlas_c,
                                        0.72, 0.72, 0.72, 0.82)

        tr.flush()

        # ── Layer 3: Alert labels (rects + text) ──
        now = datetime.now()
        self._alert_label_rects.clear()
        self._crosshair_icon_rects.clear()
        icon_half = 13
        for sys_id in list(self.manager.red_crosshair_ids):
            if sys_id not in self.manager.solar_systems:
                continue
            system = self.manager.solar_systems[sys_id]
            screen_pos = self._project_to_screen(system.xyz)
            if screen_pos is not None:
                sx, sy = int(screen_pos[0]), int(screen_pos[1])
                if 0 <= sx <= w and 0 <= sy <= h:
                    self._crosshair_icon_rects[sys_id] = QRect(
                        sx - icon_half, sy - icon_half, icon_half * 2, icon_half * 2)

        if self._camera_distance < 2000 and self._show_alert_age:
            tr.begin_frame(w, h)
            atlas_a = self._atlas_alert
            apad_x, apad_y = 6, 3
            nearby_ids = set(sid for sid in self._char_locations.values() if sid >= 0)
            nearby_ids.update(gid for gid in self.manager.green_crosshair_ids if gid >= 0)
            for sys_id in list(self.manager.red_crosshair_ids):
                if sys_id not in self.manager.solar_systems:
                    continue
                system = self.manager.solar_systems[sys_id]
                stats = self.manager.get_system_stats(sys_id)
                if not stats or stats.expired:
                    continue
                screen_pos = self._project_to_screen(system.xyz)
                if screen_pos is None:
                    continue
                sx, sy = screen_pos
                if not (0 <= sx <= w and 0 <= sy <= h):
                    continue
                parts = [system.name]
                if self._show_alert_age:
                    delta = now - stats.last_report
                    total_secs = int(delta.total_seconds())
                    mins, secs = divmod(total_secs, 60)
                    parts.append(f"({mins}:{secs:02d})")
                min_jumps = -1
                for nid in nearby_ids:
                    if nid == sys_id:
                        min_jumps = 0
                        break
                    cache_key = (nid, sys_id)
                    if cache_key in self._jump_distance_cache:
                        jumps = self._jump_distance_cache[cache_key]
                    else:
                        path = self.manager.find_path(nid, sys_id)
                        jumps = path.total_jumps if path else -1
                        self._jump_distance_cache[cache_key] = jumps
                    if jumps > 0 and (min_jumps < 0 or jumps < min_jumps):
                        min_jumps = jumps
                if min_jumps >= 0:
                    parts.append(f"{min_jumps}j")
                label = " ".join(parts)
                tw, th = atlas_a.measure_text(label)
                box_x = int(sx) + 14
                box_y = int(sy) - th // 2 - apad_y
                box_w = tw + apad_x * 2
                box_h = th + apad_y * 2
                self._alert_label_rects[sys_id] = QRect(box_x, box_y, box_w, box_h)
                tr.add_rect(box_x, box_y, box_w, box_h,
                            0.0, 0.0, 0.0, 0.75,
                            1.0, 0.28, 0.28, 0.85)
                tr.add_text(box_x + apad_x, box_y + apad_y, label, atlas_a,
                            0.72, 0.72, 0.72, 0.82)
            tr.flush()

        # ── Layer 4: Intel popup (topmost) ──
        if self._current_highlight >= 0:
            stats = self.manager.get_system_stats(self._current_highlight)
            if stats and not stats.expired and stats.last_intel_report:
                system = self.manager.solar_systems.get(self._current_highlight)
                if system:
                    screen_pos = self._project_to_screen(system.xyz)
                    if screen_pos is not None:
                        sx, sy = screen_pos
                        if 0 <= sx <= w and 0 <= sy <= h:
                            tr.begin_frame(w, h)
                            anchor_rect = self._alert_label_rects.get(self._current_highlight)
                            if anchor_rect:
                                al_box_x = anchor_rect.x()
                                al_box_bottom = anchor_rect.y() + anchor_rect.height()
                            else:
                                al_box_x = int(sx) + 14
                                al_box_bottom = int(sy) + 14

                            text = stats.last_intel_report
                            if len(text) > 100:
                                text = text[:100] + "..."
                            if not anchor_rect:
                                delta = now - stats.last_report
                                total_secs = int(delta.total_seconds())
                                mins, secs = divmod(total_secs, 60)
                                header = system.name
                                header += f" ({mins}:{secs:02d})"
                                nearby_ids = set(sid for sid in self._char_locations.values() if sid >= 0)
                                nearby_ids.update(gid for gid in self.manager.green_crosshair_ids if gid >= 0)
                                min_jumps = -1
                                for nid in nearby_ids:
                                    if nid == self._current_highlight:
                                        min_jumps = 0
                                        break
                                    cache_key = (nid, self._current_highlight)
                                    if cache_key in self._jump_distance_cache:
                                        jumps = self._jump_distance_cache[cache_key]
                                    else:
                                        path = self.manager.find_path(nid, self._current_highlight)
                                        jumps = path.total_jumps if path else -1
                                        self._jump_distance_cache[cache_key] = jumps
                                    if jumps > 0 and (min_jumps < 0 or jumps < min_jumps):
                                        min_jumps = jumps
                                if min_jumps >= 0:
                                    header += f" {min_jumps}j"
                                text = f"{header} | {text}"

                            atlas_p = self._atlas_popup
                            pp = 5
                            max_w = 220
                            lines = self._wrap_text(text, atlas_p, max_w)
                            line_h = atlas_p.line_height
                            text_block_h = line_h * len(lines)
                            max_line_w = 0
                            for line in lines:
                                lw, _ = atlas_p.measure_text(line)
                                if lw > max_line_w:
                                    max_line_w = lw
                            box_w = max_line_w + pp * 2
                            box_h = text_block_h + pp * 2
                            bx = al_box_x
                            by = al_box_bottom + 2
                            bx = max(2, min(bx, w - box_w - 2))
                            by = max(2, min(by, h - box_h - 2))
                            tr.add_rect(bx, by, box_w, box_h,
                                        0.0, 0.0, 0.0, 0.82,
                                        1.0, 0.35, 0.35, 0.90)
                            for i, line in enumerate(lines):
                                tr.add_text(bx + pp, by + pp + i * line_h, line, atlas_p,
                                            0.72, 0.72, 0.72, 0.84)
                            tr.flush()

    @staticmethod
    def _wrap_text(text: str, atlas: FontAtlas, max_w: int) -> list[str]:
        """Simple word-wrap for GPU text (no QFontMetrics needed)."""
        words = text.replace('\n', ' \n ').split(' ')
        lines: list[str] = []
        current = ""
        for word in words:
            if word == '\n':
                lines.append(current)
                current = ""
                continue
            test = (current + " " + word).strip() if current else word
            tw, _ = atlas.measure_text(test)
            if tw > max_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        return lines if lines else [""]

    def _project_to_screen(self, world_pos: np.ndarray):
        """Project a 3D world position to 2D screen coordinates."""
        pos4 = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)

        clip = self._projection @ self._modelview @ pos4
        if abs(clip[3]) < 1e-6:
            return None

        ndc = clip[:3] / clip[3]

        sx = (ndc[0] * 0.5 + 0.5) * self.width()
        sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height()

        if ndc[2] < -1 or ndc[2] > 1:
            return None

        return (sx, sy)

    # --- Matrix helpers (pure numpy, no pyrr dependency for these basics) ---

    @staticmethod
    def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
        import math
        fov_rad = math.radians(fov_deg)
        f = 1.0 / math.tan(fov_rad / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    @staticmethod
    def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = target - eye
        f_len = np.linalg.norm(f)
        if f_len > 0:
            f = f / f_len

        s = np.cross(f, up)
        s_len = np.linalg.norm(s)
        if s_len > 0:
            s = s / s_len

        u = np.cross(s, f)

        m = np.eye(4, dtype=np.float32)
        m[0, 0] = s[0]
        m[0, 1] = s[1]
        m[0, 2] = s[2]
        m[1, 0] = u[0]
        m[1, 1] = u[1]
        m[1, 2] = u[2]
        m[2, 0] = -f[0]
        m[2, 1] = -f[1]
        m[2, 2] = -f[2]
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m

    # --- Mouse events ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_x = event.position().x()
            self._start_y = event.position().y()
        elif event.button() == Qt.MouseButton.RightButton:
            picked = self._pick_system(int(event.position().x()), int(event.position().y()))
            self.system_right_clicked.emit(picked, event.globalPosition().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                dx = abs(event.position().x() - self._start_x)
                dy = abs(event.position().y() - self._start_y)
                if dx < 3 and dy < 3:
                    # It was a click, not a drag
                    picked = self._pick_system(int(event.position().x()), int(event.position().y()))
                    self._clicked_system = picked
                    if picked >= 0:
                        self.system_clicked.emit(picked)
            self._dragging = False

    def mouseMoveEvent(self, event):
        if self._dragging:
            dx = event.position().x() - self._start_x
            dy = event.position().y() - self._start_y
            scale = self._camera_distance / 500.0
            self._look_at[0] -= dx * scale
            self._look_at[1] += dy * scale
            self._start_x = event.position().x()
            self._start_y = event.position().y()
            self.update()
        else:
            # Hover detection - check alert label rects, then icon rects, then star picking
            mx, my = int(event.position().x()), int(event.position().y())
            picked = -1
            for sys_id, rect in self._alert_label_rects.items():
                if rect.contains(mx, my):
                    picked = sys_id
                    break
            if picked < 0:
                for sys_id, rect in self._crosshair_icon_rects.items():
                    if rect.contains(mx, my):
                        picked = sys_id
                        break
            if picked < 0:
                picked = self._pick_system(mx, my)
            if picked != self._current_highlight:
                self._current_highlight = picked
                if picked >= 0 and picked in self.manager.solar_systems:
                    self.system_hovered.emit(picked, self.manager.solar_systems[picked].name)
                self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_speed = self._camera_distance * 0.1
        if delta > 0:
            self._camera_distance -= zoom_speed
        else:
            self._camera_distance += zoom_speed
        self._camera_distance = max(10.0, min(15000.0, self._camera_distance))
        self.update()

    def _pick_system(self, mouse_x: int, mouse_y: int) -> int:
        """Find system under mouse cursor using ray-sphere intersection."""
        ray = MouseRay(mouse_x, mouse_y, self._modelview, self._projection,
                       (self.width(), self.height()))

        pick_radius = max(5.0, self._point_size * 2.0)
        best_dist = float('inf')
        best_id = -1

        for sys_id, system in self.manager.solar_systems.items():
            dist = ray.intersection(system.xyz, pick_radius)
            if dist > 0 and dist < best_dist:
                best_dist = dist
                best_id = sys_id

        return best_id

    def zoom_to_system(self, system_id: int):
        """Animate camera to centre on a system."""
        if system_id not in self.manager.solar_systems:
            return
        system = self.manager.solar_systems[system_id]
        self._look_at[0] = system.xf
        self._look_at[1] = system.yf
        self._camera_distance = min(self._camera_distance, 2000.0)
        self.update()

    def pan_to_system(self, system_id: int):
        """Move camera to centre on a system without changing zoom."""
        if system_id not in self.manager.solar_systems:
            return
        system = self.manager.solar_systems[system_id]
        self._look_at[0] = system.xf
        self._look_at[1] = system.yf
        self.update()

    def center_on_position(self, x: float, y: float):
        self._look_at[0] = x
        self._look_at[1] = y
        self.update()
