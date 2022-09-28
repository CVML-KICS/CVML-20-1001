
import ray
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from HelperFunctionParking import *
from scipy.spatial import distance
from itertools import takewhile
import multiprocessing as mp
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
# Number of classes the object detector can identify
NUM_CLASSES = 6
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.





# Open video file



#p = mp.Process(target=run)
#p.start()

nxt=1
collidedRecs = []
indexes = []
anyRecContain = False
i = 0
CL1=[]
CL2=[]
prevCL1={}
MovingThreashold=15
index=0
def CalculateDiff(prevCL1):
    return distance.euclidean((prevCL1[0][0],prevCL1[0][1]),(prevCL1[len(prevCL1)-1][0],prevCL1[len(prevCL1)-1][1]))

@ray.remote(num_gpus=0.38)
class TrainingActor(object):
    def __init__(self, seed):
        print('Set new seed:', seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.38)
        #self.mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

        # # Setting up the softmax architecture.
        # self.x = tf.placeholder('float', [None, 784])
        # W = tf.Variable(tf.zeros([784, 10]))
        # b = tf.Variable(tf.zeros([10]))
        # self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)
        #
        # # Setting up the cost function.
        # self.y_ = tf.placeholder('float', [None, 10])
        # cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        # Initialization

        self.init = tf.initialize_all_variables()
        self.sess1 = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess1 = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, inter_op_parallelism_threads=2,
                intra_op_parallelism_threads=2), graph=self.detection_graph)

    
    def train(self,path, name):

        video = cv2.VideoCapture(path)
        global gpu_options
        global detection_graph


        global index
        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        while (video.isOpened()):
            print(name)

            ret, frame = video.read()
            frame = cv2.resize(frame,(720,480),frame, interpolation=cv2.INTER_AREA)

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value

            # if index%2==0:
            frame_expanded = np.expand_dims(frame, axis=0)

            # Perform the actual detection by running the model with the image as input
            #

            (boxes, scores, classes, num) = self.sess1.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            index = index + 1
            frame, boxes, labels = vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.70)
            # All the results have been drawn on the frame, so it's time to display it.
            labelsName = ['car', 'van', 'truck', 'bus', 'motorcycle', 'rickshaw']
            labelCount = []
            displayString = ''
            for label in labelsName: count = len(
                [x for x in list(labels.values()) if len(x) > 0 and label in x[0]]);labelCount.append(count);
            resizedBoxes = [(x[1] * frame.shape[1], x[0] * frame.shape[0], x[3] * frame.shape[1], x[2] * frame.shape[0])
                            for
                            x in boxes]
            # frame1 = cv2.resize(frame,None,fx=0.60, fy=0.60, interpolation=cv2.INTER_AREA)
            frame1 = frame
            i = 0
            TotalMoving=0

            for rect in resizedBoxes:
                showIndex = 0
                MovingStatus = "Static"

                if len(CL1) == 0:
                    CL1.append(rect)
                    prevCL1[len(CL1) - 1] = []
                    showIndex = i
                    i = i + 1
                    indexes.append(i)


                else:
                    index, found = CheckIOU(CL1, rect)
                    if not found:
                        CL1.append(rect)
                        prevCL1[len(CL1) - 1] = []
                        showIndex = i
                        i = i + 1
                        indexes.append(i)

                    elif found:
                        if len(prevCL1[index]) <= 10:
                            prevCL1[index].append(rect)
                        else:
                            prevCL1[index].pop(0)
                            prevCL1[index].append(rect)

                        if (CalculateDiff(prevCL1[index]) > MovingThreashold):
                            MovingStatus = "Moving"
                            TotalMoving+=1

                        CL1[index] = rect
                        indexes.append(index)
                        showIndex = index

                cv2.rectangle(frame1, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), color=(200, 200, 200),
                              thickness=1)
                cv2.putText(frame1, str(showIndex), (int(rect[0]), int(rect[1]) - 10), 1, 1, (0, 0, 0))
                cv2.putText(frame1, str(MovingStatus), (int(rect[0]) + 10, int(rect[1] - 10) - 10), 1, 1, (0, 0, 0))

            cv2.rectangle(frame1, (0, 0), (150, 200), color=(200, 200, 200), thickness=-1)
            y = 5
            for i in range(len(labelsName)):
                y = y + 20
                displayString = labelsName[i] + ':' + str(labelCount[i])
                cv2.putText(frame1, displayString, (5, y), 1, 1, (0, 0, 0))
            cv2.putText(frame1, 'Moving: '+  str(TotalMoving), (5, y+15), 1, 1, (0, 0, 0))
            cv2.putText(frame1, 'Static: ' + str(len(resizedBoxes)-TotalMoving), (5, y + 35), 1, 1, (0, 0, 0))
            cv2.imshow(name, frame1)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break


        ############################################################33


        return path
if __name__ == '__main__':
    # Start Ray.
    ray.init(num_gpus=1)
    path=[]
    name=[]
    path.append('VehicleSimulation.mp4')
    name.append('video1')
    path.append('LahoreSafeCity.mp4')
    name.append('video2')
    b=0

    # Create 3 actors.
    training_actors = [TrainingActor.remote(seed) for seed in range(2)]

    # Make them all train in parallel.
    #accuracy_ids = [actor.train.remote() for actor in training_actors]
    accuracy_ids = {i + 1: [training_actors[i].train.remote(path[i],name[i])] for i in range(2)}

    print(ray.get(accuracy_ids[1]))
    print(ray.get(accuracy_ids[2]))

    # Start new training runs in parallel.
    #accuracy_ids = [actor.train.remote('abc') for actor in training_actors]
    #print(ray.get(accuracy_ids))
