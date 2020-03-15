import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from face_tools import predictor

detector = MTCNN()
model = VGGFace(model='senet50',include_top=False, input_shape=(224, 224, 3), pooling='max')
font = cv2.FONT_HERSHEY_SIMPLEX
line = cv2.LINE_AA
saved_face = predictor('registered_img.jpg')


def verification():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        result = detector.detect_faces(frame)
        if result != [] and result[0]['confidence'] > 0.90:
            x1, y1, w, h = result[0]['box']
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = x1 + w
            y2 = y1 + h
            crop = frame[y1:y2, x1:x2]
            face = cv2.resize(crop, (224, 224))

            start_pt = (x1, y1)
            end_pt = (x2, y2)

            face = np.asarray(face, 'float32')
            face = preprocess_input([face], version=2)
            prediction = model.predict(face)[0]
            cos_distance = round(cosine(saved_face, prediction), 2)
            box = ('Matching', (0, 255, 0)) if cos_distance < 0.4 else ('Not Matching', (255, 0, 0))

            cv2.rectangle(frame, start_pt, end_pt, thickness=2, color=box[1])
            cv2.putText(frame, box[0], (x1 - 5, y1 - 6), font, 0.7, box[1])

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

verification()


