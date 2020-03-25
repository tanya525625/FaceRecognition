import cv2
import os


def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    return gray_img[y:y + w, x:x + h], faces[0]
