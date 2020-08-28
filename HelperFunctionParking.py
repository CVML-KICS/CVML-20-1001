from shapely.geometry import Point, Polygon
def rectContains(rect,pt):
    logic = rect[0] < pt[0] < rect[2] and rect[1] < pt[1] < rect[3]
    return logic


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
    if threashold:
    	currentThreashHold=0.8
    index=-1
    i=0
    found=False
    for curRect in CL:
        if(bb_intersection_over_union(curRect,rect)>currentThreashHold):
            found=True
            index=i
        i=i+1
    return index,found
def CheckIOUList(CL,rect,threashold=False):
    collidedRects=[]
    currentThreashHold=0.05
    if threashold:
    	currentThreashHold=0.8
    index=-1
    i=0
    found=False
    for curRect in CL:
        if(bb_intersection_over_union(curRect,rect)>currentThreashHold):
            found=True

            index=i
            collidedRects.append(index)
        i=i+1
    return collidedRects,found

def contains(rect, points):
    index=-1
    found=False
    i=0
    for p in points:
        if rectContains(rect,p):
            index=i
            found=True
        i=i+1
    return found
def findCollessionwithRect(rects,linePoints,CL):
    collidedRecs=[]
    indexes=[]
    anyRecContain=False
    i=0

    for rect in rects:

        if contains(rect,linePoints):
            anyRecContain=True
            if len(CL)==0:
                CL.append(rect)
                collidedRecs.append(rect)
                indexes.append(i)
            else:
                index,found=CheckIOU(CL,rect)
                if not found:
                    CL.append(rect)
                    collidedRecs.append(rect)
                    indexes.append(i)
                elif found:
                    CL[index]=rect
        else:
            index,found=CheckIOU(CL,rect,True)
            if found:
               del  CL[index]
        i=i+1
    return collidedRecs,indexes,anyRecContain
