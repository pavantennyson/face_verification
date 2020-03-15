import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


def Registration():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        result = detector.detect_faces(frame)
        if result != []:
            start_pt = (result[0]['box'][0], result[0]['box'][1])
            end_pt = (result[0]['box'][0] + result[0]['box'][2], result[0]['box'][1] + result[0]['box'][3])
            cv2.rectangle(frame, start_pt, end_pt, thickness=1, color=(0, 255, 0))

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            cv2.imwrite('registered_img.jpg', frame)
            cap.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


Registration()