import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.prev_time = time.time()  # Tiempo previo para cálculo de FPS

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convertir BGR a RGB para Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesamiento según el modo seleccionado
            if self.mode == "angle":
                self.process_angle(image, frame)
            elif self.mode == "face":
                self.process_face_mesh(image, frame)
            elif self.mode == "hand":
                self.process_hand(image, frame)

            # Calcular FPS
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            # Mostrar FPS en la esquina superior izquierda
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convertir a QImage para enviar a la interfaz
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap_signal.emit(q_img)

    def process_angle(self, image, frame):
        results = self.mp_pose.process(image)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            angle = self.calculate_angle(shoulder, elbow, wrist)
            cv2.putText(frame, f"Ángulo del codo: {int(angle)}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    def process_face_mesh(self, image, frame):
        results = self.mp_face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_LIPS)
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
                self.mp_drawing.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)

    def process_hand(self, image, frame):
        results = self.mp_hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def calculate_angle(self, point1, point2, point3):
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()

class AsyncElbowAngleApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurar la interfaz
        self.setWindowTitle("Cálculo de Ángulo de Codo, Detección de Rostro y Dedos con Mediapipe")
        self.setGeometry(200, 200, 800, 600)

        # Crear layout y widgets
        self.video_label = QLabel()
        self.angle_button = QPushButton("Ángulo de Codo")
        self.angle_button.clicked.connect(lambda: self.set_mode("angle"))
        self.face_button = QPushButton("Face Mesh")
        self.face_button.clicked.connect(lambda: self.set_mode("face"))
        self.hand_button = QPushButton("Detección de Dedos")
        self.hand_button.clicked.connect(lambda: self.set_mode("hand"))
        self.stop_button = QPushButton("Detener Cámara")
        self.stop_button.clicked.connect(self.stop_video_thread)
        self.exit_button = QPushButton("Salir")
        self.exit_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.angle_button)
        layout.addWidget(self.face_button)
        layout.addWidget(self.hand_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.exit_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Hilo de video
        self.video_thread = None
        self.mode = None

    def set_mode(self, mode):
        self.mode = mode
        if self.video_thread is None or not self.video_thread.isRunning():
            self.start_video_thread()

    def start_video_thread(self):
        self.video_thread = VideoThread(self.mode)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def stop_video_thread(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.video_label.clear()

    def update_image(self, q_img):
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.stop_video_thread()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AsyncElbowAngleApp()
    window.show()
    sys.exit(app.exec())
