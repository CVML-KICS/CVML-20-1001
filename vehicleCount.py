
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import rtsp
import cv2 as cv
from shapely.geometry import Point, Polygon
from HelperFunctionParking import*
from ast import literal_eval

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from .utils import label_map_util
from .utils import visualization_utils as vis_util

from itertools import takewhile
# Name of the directory containing the object detection module we're using

MODEL_NAME = 'inference_graph'
VIDEO_NAME = '/home/lodhi/Videos/vehicles/test.avi'

Lineboxes=[]
CL1=[]
CL2=[]
def findPoints(p1,p2):
    if (p1 is not None and p2 is not None):
        Checked = True
        Infinite = False
        if p2.x == p1.x:
            Infinite = True
        else:
            m1 = (p2.y - p1.y) / (p2.x - p1.x)
            # b = p1.y - (m1 * p1.x)

        rangeDiff_X = int(abs(p1.x - p2.x))
        rangeDiff_Y = int(abs(p1.y - p2.y))
        Current_X = p1.x if p1.x < p2.x else p2.x
        Current_Y = p1.y if p1.y < p2.y else p2.y
        # Current_Y = Current_Y + 1
        pointList = []
        for i in range(rangeDiff_X):
            Current_Y = p1.y if p1.y < p2.y else p2.y
            for j in range(rangeDiff_Y):
                p3 = Point(Current_X, Current_Y)
                m2 = None
                if not Infinite and p3.x == p1.x:
                    Current_Y = Current_Y + 1
                    continue
                elif not Infinite and not (p3 == p1 or p3 == p2):
                    m2 = (p3.y - p1.y) / (p3.x - p1.x)
                if (m2 is not None and m1 is not None and round(m1, 1) == round(m2, 1)) \
                        or (Infinite and p3.x == p1.x) \
                        or (p3 == p1 or p3 == p2):
                    pointList.append((int(p3.x), int(p3.y)))
                Current_Y = Current_Y + 1
            Current_X = Current_X + 1
    return pointList
def on_mouse(event, x, y, flags, params):
    # global img

    if event == cv2.EVENT_FLAG_LBUTTON:
        global Lineboxes
        if(len(Lineboxes)>4):
            Lineboxes= []

        print ('Start Mouse Position: ' + str(x) + ', ' + str(y))
        sbox = (x,y)
        Lineboxes.append(sbox)
        # print count
        # print sbox

    elif event == cv2.EVENT_LBUTTONUP:
        print ('End Mouse Position: ' + str(x) + ', ' + str(y))
        ebox = (x,y)
        Lineboxes.append(ebox)


def intersection(self, other,ver):
    a, b = self, other
    m= (b[1][1]-b[0][1])/(b[1][0]-b[0][0])

    p1 = Point(int((a[0]+a[2])/2),int((a[1]+a[3])/2))
    # Create a Polygon
    if not ver:
        coords = [(b[0][0], b[0][1]), (b[0][0], b[0][1]+7), (b[1][0], b[1][1]+7),(b[1][0], b[1][1])]
    else:
        coords = [(b[0][0], b[0][1]), (b[0][0]+7, b[0][1]), (b[1][0]+7, b[1][1]), (b[1][0], b[1][1])]
    poly = Polygon(coords)
    return p1.within(poly)


def findCollession(rects,line,ver=False):
    collidedRecs=[]
    indexes=[]
    i=0
    for rect in rects:

        if intersection(rect,line,ver):
            collidedRecs.append(rect)
            indexes.append(i)
        i=i+1
    return collidedRecs,indexes


def findCollessionwithLine(rects,linePoints,ver=False):
    collidedRecs=[]
    indexes=[]
    i=0

    for rect in rects:
        a=rect
        p1 = Point(int(a[2]) , int(a[1]))

        if (p1.x,p1.y) in linePoints:
            collidedRecs.append(rect)
            indexes.append(i)
        i=i+1
    return collidedRecs,indexes
# Grab path to current working directory
CWD_PATH = os.getcwd()
CWD_PATH='/home/lodhi/PycharmProjects/vehicle_api/VehicleTypeCount/vehicle_count/object_detection'
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
import sys
sys.path.append('/home/lodhi/PycharmProjects/vehicle_api/VehicleTypeCount/vehicle_count/object_detection/training')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.


# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

nxt=1

labelsName=['car','van','truck','bus','motorcycle','rickshaw']
labelCount=0
labelCount2=0
Total_capacity=100
Vacant_Space=50
BetweenPoints1=None
BetweenPoints2=None
loaded=False
sess=None
detection_boxes=None
detection_scores=None
detection_classes=None
num_detections=None
image_tensor=None
def countVehicle(frame):
    global loaded
    global sess
    global detection_boxes
    global detection_scores
    global detection_classes
    global num_detections
    global image_tensor
    global Lineboxes
    if not loaded:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        cv2.namedWindow('Object detector')
        cv2.setMouseCallback('Object detector', on_mouse)
        loaded=True


    global labelsName
    global labelCount
    global labelCount2
    global Total_capacity
    global Vacant_Space
    global BetweenPoints1
    global BetweenPoints2
    #controls which one frame to be proceed next
    #nxt=nxt+2
    #video.set(1,nxt)

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value


    frame = cv2.resize(frame, (720,720), interpolation=cv2.INTER_AREA)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    frame1 = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    frame=frame1

    # Draw the results of the detection (aka 'visulaize the results')
    frame, boxes,labels=vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.99)
    # All the results have been drawn on the frame, so it's time to display it.



    displayString=''
    #ymin, xmin, ymax, xmax
    resizedBoxes=[(x[1] * frame.shape[1], x[0] * frame.shape[0], x[3] * frame.shape[1], x[2] * frame.shape[0]) for x in boxes]


    #for label in labelsName: count = len([x for x in list(labels.values()) if len(x) > 0 and label in x[0]]);labelCount.append(count);





    cv2.rectangle(frame1,(0,0),(350,50),color=(200,200,200), thickness=-1)


    if len(Lineboxes)>1:
        collidedRects2=[]
        cv2.line(frame1, Lineboxes[0], Lineboxes[1], (255, 0, 0), 5)
        if BetweenPoints1 is None:

            BetweenPoints1=findPoints(Point(Lineboxes[0][0],Lineboxes[0][1]),Point(Lineboxes[1][0],Lineboxes[1][1]))
        for point in BetweenPoints1:
            cv2.rectangle(frame1,point,(point[0]+1,point[1]+1),(0, 0, 0), thickness=2)
        if len(Lineboxes)>3:
            f = open("count.txt", "w")
            f.write(str(Lineboxes))
            f.close()

        # open and read the file after the appending:
            f = open("count.txt", "r")
            Lineboxes=literal_eval(f.read())
            cv2.line(frame1, Lineboxes[2], Lineboxes[3], (255, 0, 0), 5)
            if BetweenPoints2 is None:
                BetweenPoints2 = findPoints(Point(Lineboxes[2][0], Lineboxes[2][1]),
                                        Point(Lineboxes[3][0], Lineboxes[3][1]))

            #collidedRects2, indexes2 = findCollessionwithLine(resizedBoxes, BetweenPoints2,True)
            collidedRects2, indexes2,cnt = findCollessionwithRect(resizedBoxes, BetweenPoints2,CL2)
            for point in BetweenPoints2:
            	cv2.rectangle(frame1,point,(point[0]+1,point[1]+1),(0, 0, 0), thickness=2)

        # for rect in resizedBoxes:
        #     a = rect
        #     p1 = (int(a[2]), int(a[1]))
        #     cv2.rectangle(frame1, p1, (p1[0] + 1, p1[1] + 1), (0, 0, 0), thickness=4)


        #collidedRects,indexes=findCollessionwithLine(resizedBoxes,BetweenPoints1)
        collidedRects,indexes, cont=findCollessionwithRect(resizedBoxes,BetweenPoints1,CL1)



        y = 5
        ############################ start from here ####################################3
        for collidedRectLeft in collidedRects2:

            for i in range(len(indexes2)):

                displayString2 = list(labels.values())[indexes2[i]][0][0:list(labels.values())[indexes2[i]][0].find(':')]
                indexOfvehicle2=labelsName.index(displayString2)
                labelCount2 =labelCount2+1
                Vacant_Space = Vacant_Space-1
                #cv2.rectangle(frame1, (int(collidedRects2[0]), int(collidedRects2[1])),(int(collidedRects2[2]), int(collidedRects2[3])), (0, 0, 0), thickness=20)

        for collidedRect in collidedRects:

            for i in range(len(indexes)):

                displayString = list(labels.values())[indexes[i]][0][0:list(labels.values())[indexes[i]][0].find(':')]
                indexOfvehicle=labelsName.index(displayString)
                labelCount =labelCount +1
                Vacant_Space = Vacant_Space +1
                #cv2.rectangle(frame1, (int(collidedRect[0]), int(collidedRect[1])),(int(collidedRect[2]), int(collidedRect[3])), (0, 0, 0), thickness=20)

        labeltoDisplay=" Entered Vehicles: "+ str(labelCount)
        cv2.putText(frame1, labeltoDisplay, (5, 20), 1, 1, (0, 0, 0))

        labeltoDisplay=" Exit Vehicles: "+ str(labelCount2)
        cv2.putText(frame1, labeltoDisplay, (180, 20), 1, 1, (0, 0, 0))
        result = str(labelCount) + "," + str(labelCount2)
    #print(labelCount)
    #cv2.imshow('Object detector', frame1)
    return frame1

    # Press 'q' to quit
    #if cv2.waitKey(1) == ord('q'):
        #break

# Clean up

def vehiclecount(path):
    video= cv2.VideoCapture(path)
    ret,frame=video.read()
    #client= rtsp.Client(rtsp_server_uri='rtsp://kics.uet:kics@12345@10.11.7.82:554/cam/realmonitor?channel=1&subtype=0')
    #frame = client.read()
    #print(frame)

    while ret:
      t1=time.time()
      if frame is not None:
        ret, frame = video.read()
        # ret, frame = video.read()
        cv2.imshow('Object detector', countVehicle(frame))
        if cv2.waitKey(1) == ord('q'):
            break
      print(time.time()-t1)
    video.release()
    cv2.destroyAllWindows()
    #enterData = VehicleCount(Enter=labelCount, Exit=labelCount2)
    #enterData.save()

    return labelCount,labelCount2



vehiclecount('rtsp://kics.uet:kics@12345@10.11.7.82:554/cam/realmonitor?channel=1&subtype=0')
# print("enter vehicle: " +str(x),"Exit Vehicle: "+str(y))
