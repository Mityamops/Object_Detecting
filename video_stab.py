import cv2
import numpy as np
import numpy.linalg as lin

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.08)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def stabilize(frame_color, amount: int) :

    max_n = amount
    prev_n_shift = [np.identity(3)]
    prev_frame = None


    while True:

        frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape
        prev_pts = cv2.goodFeaturesToTrack(frame, maxCorners=200,
                                           qualityLevel=0.01, minDistance=30,
                                           blockSize=3)
        if prev_frame is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                             prev_pts, None)
            M, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
            prev_n_shift.append(np.dot(M, prev_n_shift[-1]))
            if len(prev_n_shift) > max_n:
                prev_n_shift.pop(0)
                inv = lin.inv(prev_n_shift[0])
                for i in range(len(prev_n_shift)):
                    prev_n_shift[i] = np.dot(prev_n_shift[i], inv)
            prev_frame = frame
            sM = np.dot(np.average(np.array(prev_n_shift), 0), lin.inv(prev_n_shift[-1]))
            output_frame = cv2.warpPerspective(frame_color, sM, (w, h))
        else:
            prev_frame = frame
            output_frame = frame_color

        #out.write(output_frame)

        output_frame=fixBorder(output_frame)
        '''
        cv2.imshow('frame', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''
        return output_frame

