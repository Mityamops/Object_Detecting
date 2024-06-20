# import required libraries
from vidgear.gears.stabilizer import Stabilizer
import cv2

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame




# initiate stabilizer object with default parameters
stab = Stabilizer(smoothing_radius=10)

# loop over
def stabilized_frame(frame):
    stab_frame = stab.stabilize(frame)
    if stab_frame is None:
        stab_frame=frame
    else:
        stab_frame=fixBorder(stab_frame)

    return stab_frame