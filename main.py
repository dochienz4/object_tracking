import cv2
# import numpy as np
from object_detection import ObjectDetection


def _main(video_path):
    od = ObjectDetection()
    cap = cv2.VideoCapture(video_path)
    center_p = {'object_count': [], 'object_center': []}
    count = 0
    center_points = []  # all center in one frame

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        (class_id, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            c_x = int((2 * x + w) / 2)
            c_y = int((2 * y + h) / 2)
            center_points.append((c_x, c_y))

            count += 1
            center_p['object_count'].append(count)
            center_p['object_center'].append([c_x, c_y])

            # cv2.circle(img=frame, center=(c_x, c_y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        for point in center_points:
            cv2.circle(img=frame, center=point, radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0)
        if key == 27:  # ESCAPE key
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # _main('resource/video.mp4')
    _main('resource/los_angeles.mp4')
