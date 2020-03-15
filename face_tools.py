import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


def face_extractor(img_path, img_dimension=(224, 224)):
    faces = []
    img = cv2.imread(img_path)
    rectangles = detector.detect_faces(img)
    for rect in rectangles:
        if rect['confidence'] > 0.95:
            x1, y1, w, h = rect['box']
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = x1 + w
            y2 = y1 + h
            crop = img[y1:y2, x1:x2]
            face = cv2.resize(crop, img_dimension)
            faces.append(face)

    return faces


model = VGGFace(model='senet50',include_top=False, input_shape=(224, 224, 3), pooling='max')


def predictor(img):
    face = face_extractor(img)[0]
    face = np.asarray(face, 'float32')
    face = preprocess_input([face], version=2)

    predictions = model.predict(face)
    return predictions[0]



