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
video = cv2.VideoCapture('/home/crowdteam/ParkingVideo.mp4')
Checked=False
while (video.isOpened()):
    ret, frame = video.read()

    if(p1 is not None and p2 is not None) and not Checked:
        Checked=True
        Infinite=False
        if p2.x==p1.x:
            Infinite=True
        else:
            m1 = (p2.y - p1.y) / (p2.x - p1.x)
            #b = p1.y - (m1 * p1.x)

        rangeDiff_X = int(abs(p1.x - p2.x))
        rangeDiff_Y = int(abs(p1.y - p2.y))
        Current_X = p1.x if p1.x < p2.x else p2.x
        Current_Y = p1.y if p1.y < p2.y else p2.y
        #Current_Y = Current_Y + 1
        pointList = []
        for i in range(rangeDiff_X):
            Current_Y=p1.y if p1.y < p2.y else p2.y
            for j in range(rangeDiff_Y):
                p3 = Point(Current_X, Current_Y)
                m2=None
                if not Infinite and p3.x==p1.x:
                    Current_Y = Current_Y + 1
                    continue
                elif not Infinite and not (p3==p1 or p3==p2):
                    m2 = (p3.y - p1.y) / (p3.x - p1.x)
                if (m2 is not None and m1 is not None and round(m1,1) == round(m2,1))\
                        or (Infinite and p3.x==p1.x)\
                        or (p3==p1 or p3==p2):
                    pointList.append((int(p3.x),int(p3.y)))
                Current_Y=Current_Y+1
            Current_X=Current_X+1
    elif Checked:
        #cv2.line(frame, Lineboxes[0], Lineboxes[1], (255, 0, 0), 5)
        for point in pointList:
            cv2.rectangle(frame,point,(point[0]+1,point[1]+1),(0, 0, 0), thickness=1)
    if ret:
        cv2.imshow('Object detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break

len(pointList)


