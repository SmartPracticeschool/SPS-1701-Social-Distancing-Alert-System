from .social_distancing_configuration import MIN_CONFIDENCE ,NMS_THRESHOLD
import numpy as np
import cv2

def detect_people(frame, net, layerNames, personIdx=0):
    #Grab the dimensions of the frame and initialize the list results
    (H, W) = frame.shape[:2]
    results = []

    #Preprocessing the frames
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    #Give the preprocessed frames into the network
    net.setInput(blob)

    #Takes the output from the network
    outputLayers = net.forward(layerNames)

    #Initialize bounding boxes, centroids and confidences
    boxes = []
    centroids = []
    confidences = []

    for output in outputLayers:
        for detection in output:
            #Yolo model gives 8 outputs - confidence, x, y, h, w, pr(1), pr(2), pr(3)
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    #Non Maxima Supression to suppress the weak and overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idxs) > 0:
        #Loop over the indexes we are keeping
        for i in idxs.flatten():
            #Extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #Update the results list
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    #Return the list of results - Centroid, Bounded box and Confidence
    return results
