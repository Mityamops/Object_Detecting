import numpy as np
import cv2
import Tracker

# filename = 'http://192.168.217.103/mjpg/video.mjpg'


filename = 'videos/output_video.mp4'


def resizing(frame):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim)


percent = 30

cap = cv2.VideoCapture(filename)

'''
suc, frame = cap.read()
frame = resizing(frame)
suc, frame_next = cap.read()
frame_next = resizing(frame_next)
'''

frame_list = []
for i in range(10):
    suc, n_frame = cap.read()
    n_frame = resizing(n_frame)
    frame_list.append(n_frame)

frame = frame_list[0]
frame_next = frame_list[1]

width = int(frame.shape[1])
height = int(frame.shape[0])

iter_for_dil = 15
max_distance = 100  # максимальное расстояние на которое может переместиться обьект за один кадр
area_treshhold = width * height * 1 / 2000  # минимально допустимая площадь для отрисовки контура
distance_traectory = 200  # допустимое расстояние между двумя точками , для отслеживания трека
trajectory_len = 40  # длина траектории
detect_interval = 10  # раз в сколько кадров обновляем траектории
trajectories = []
frame_idx = 0  # счетчик кадров

tracker = Tracker.EuclideanDistTracker(max_distance)


def Object_Detect(frame, frame_next, area_tresh):
    mean = np.median(frame_list, axis=0).astype(dtype=np.uint8)
    frame=np.median(frame_list[4:], axis=0).astype(dtype=np.uint8)
    diff = cv2.absdiff(frame, mean)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 25)

    _, mask_obj = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    blur_contours, _ = cv2.findContours(mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_blur = np.zeros_like(blur)
    for cnt in blur_contours:
        if len(cnt) > 1:
            cv2.drawContours(new_blur, cnt, -1, 255, 2)

    dilated = cv2.dilate(new_blur, None, iterations=iter_for_dil)
    erode = dilated
    # erode = cv2.erode(dilated, None, iterations=7)

    contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_with_area = []
    detections = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        cont_with_area.append((cnt, area_cnt))

    cont_with_area.sort(key=lambda x: x[1], reverse=True)
    for cnt, area in cont_with_area:

        # Calculate area and remove small elements

        if area > area_tresh:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    upd = tracker.update(detections)
    return upd, mask_obj


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
                      qualityLevel=0.9,
                      minDistance=10,
                      blockSize=7)

id_list = []

while True:
    img = frame.copy()

    boxes_ids, dilat = Object_Detect(frame, frame_next, area_treshhold)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    frame_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
    if len(trajectories) > 0:

        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < distance_traectory
        p1_new = p1.reshape(-1, 2)

        new_trajectories = []

        # Get all the trajectories
        for trajectory, (x, y), good_flag in zip(trajectories, p1_new, good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            # Newest detected point
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        cv2.polylines(frame, [np.int32(trajectory) for trajectory in trajectories], False, (0, 0, 255), 2)
        # cv2.putText(frame, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track

        for box_id in boxes_ids:
            x, y, w, h, id = box_id

            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            if (x + w) > cx > x and (y + h) > cy > y:
                if not (id in id_list):
                    trajectories.append([(cx, cy)])
                    id_list.append(id)

    frame_idx += 1

    cv2.imshow('Optical Flow', frame)
    cv2.imshow('dilat', dilat)
    # cv2.imshow('Mask', mask)

    frame_list.remove(frame)
    frame = frame_next  #
    suc, new_frame = cap.read()
    new_frame = resizing(new_frame)
    frame_list.append(new_frame)

    frame_next = frame_list[1]
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()