import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront
from pywavefront.visualization import draw
from PIL import Image
import os

class ModelWidget(QOpenGLWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(16)  # Aproximadamente 60 FPS

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        self.load_model()

    def load_model(self):
        self.scene = pywavefront.Wavefront(self.model_path, collect_faces=True)
        self.textures = {}
        for name, material in self.scene.materials.items():
            if material.texture:
                texture_path = material.texture.path
                self.textures[name] = self.load_texture(texture_path)

    def load_texture(self, texture_path):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        img = Image.open(texture_path)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert("RGBA").tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture_id

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(self.angle, 1.0, 1.0, 1.0)
        for name, material in self.scene.materials.items():
            if name in self.textures:
                glBindTexture(GL_TEXTURE_2D, self.textures[name])
            else:
                glBindTexture(GL_TEXTURE_2D, 0)
            draw(self.scene, name)

    def update_angle(self):
        self.angle += 1
        self.update()

class MainWindow(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Visualizador de Modelo 3D")
        self.setGeometry(100, 100, 800, 600)
        self.model_widget = ModelWidget(model_path)
        self.setCentralWidget(self.model_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, "Rubik's Cube.obj")
    window = MainWindow(model_path)
    window.show()
    sys.exit(app.exec_())
