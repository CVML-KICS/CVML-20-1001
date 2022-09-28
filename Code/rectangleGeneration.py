import os
import cv2
from shapely.geometry import Point, Polygon

p1 = None
p2 = None
Lineboxes = []


def on_mouse(event, x, y, flags, params):
    # global img

    if event == cv2.EVENT_FLAG_LBUTTON:
        global Lineboxes
        global p1
        if (len(Lineboxes) > 4):
            Lineboxes = []

        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        p1 = Point(x, y)
        Lineboxes.append((x, y))
        # print count
        # print sbox

    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: ' + str(x) + ', ' + str(y))
        global p2
        p2 = Point(x, y)
        Lineboxes.append((x, y))


cv2.namedWindow('Object detector')
cv2.setMouseCallback('Object detector', on_mouse)
video = cv2.VideoCapture('/home/abdullah/SignalTest.mp4')
Checked=False
rectangles=[]
while (video.isOpened()):
    ret, frame = video.read()

    if(p1 is not None and p2 is not None):
        rectangles.append((p1,p2))
        p1=None
        p2=None
    if ret:
        for rect in rectangles:

            cv2.rectangle(frame, (int(rect[0].x),int(rect[0].y)), (int(rect[1].x),int(rect[1].y)), (0, 255, 0), thickness=2)

        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break




