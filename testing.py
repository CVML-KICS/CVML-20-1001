def trck(resized_boxes,CL1):
    indexes = []
    i=0
    for rect in resized_boxes:
                showIndex = 0

                if len(CL1) == 0:
                    CL1.append(rect)
                    #prevCL1[len(CL1) - 1] = []
                    showIndex = i

                    indexes.append(i)
                    i = i + 1


                else:
                    index, found = CheckIOU(CL1, rect)
                    if not found:
                         CL1.append(rect)
                         #prevCL1[len(CL1) - 1] = []
                         showIndex = i
                         indexes.append(i)
                         i = i + 1

                    elif found:
                        CL1[index] = rect
                        indexes.append(index)
                        showIndex = index
    return indexes




def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
def CheckIOU(CL,rect,threashold=False):
    currentThreashHold=0.2
    if threashold:currentThreashHold=0.8
    index=-1
    i=0
    found=False
    for curRect in CL:
        if(bb_intersection_over_union(curRect,rect)>currentThreashHold):
            found=True
            index=i
        i=i+1
    return index,found


