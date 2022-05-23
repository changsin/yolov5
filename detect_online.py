import threading
import time

import cv2
import yolov5

import detect

"""
"""
def detect_objects():

    detect.run(weights='models/yolov5s.pt', source='0', data='data/coco128-vi.yaml')

    model = yolov5.load('models/yolov5s.pt')

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        detected = model(frame)
        detected.save("results")

        cv2.imshow('YOLO', detected.imgs[0])

        c = cv2.waitKey(1)
        # Press 'q' to quit
        if c == 113:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_objects()