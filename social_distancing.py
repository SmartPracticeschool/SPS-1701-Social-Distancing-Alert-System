from packages import social_distancing_configuration as configure
from packages.object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os

#Give path to the class labels on which the YOLO model was trained on
labelsPath = os.path.sep.join([configure.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#Give the paths to the YOLO Weights and Model configuration
weightsPath = os.path.sep.join([configure.MODEL_PATH, "yolov3.weights"])
configurePath = os.path.sep.join([configure.MODEL_PATH, "yolov3.cfg"])

#Load our YOLO object detector trained on COCO dataset(COC0 80 Classes)
print("[INFO] Loading YOLO from the disk...")

#Initialize the network
net = cv2.dnn.readNetFromDarknet(configurePath, weightsPath)

#Check if GPU is being used or not
if configure.USE_GPU:
    #Set CUDA as preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Determining only those output layer names that we need from YOLO
layerNames = net.getLayerNames()
layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Initializing the video stream and pointer to output video file
print("[INFO] accesseing video stream...")
#For Camera as input
# cap = cv2.VideoCapture(0) 
cap = cv2.VideoCapture(r"pedestrian.mp4" if "pedestrian.mp4" else 0 )

#Initialize our output video writer to None
writer = None

#Loop over the frames of the video/live stream
while True:
    #Read the file
    (grabbed, frame) = cap.read()

    #If the frame not grabbed, then exit the loop
    if not grabbed:
        break
    
    #Resize the frame and then detect only people in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, layerNames, personIdx=LABELS.index("person"))
    
    #Initialize the set of indexes that voilate the minimum social distance
    voilate = set()
    
    if len(results) >= 2:
        #Extract the centroids from the results 
        centroids = np.array([r[2] for r in results])
        #Compute the Euclidean distances between pair of the centroids
        D = dist.cdist(centroids, centroids, metric = "euclidean")
        
        #loop over the upper triangular matrix of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                #checking if the distance between any two centroid pairs
                #is less than the MIN_DIST = configured no.of pixels
                
                if D[i, j] < configure.MIN_DISTANCE:
                    #Update the voilation set 
                    voilate.add(i)
                    voilate.add(j)

    #Loop over the results 
    for (i, (prob, bbox, centroid)) in enumerate(results):
        #Extract the Bounding Box and centroid coordinates, 
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        
        #Initialize the color of the annotation (in BGR)
        #Initially GREEN color all the persons 
        color = (0, 255, 0)
        
        #If the index pair exists within the violation set, then update their colour to RED
        if i in voilate:
            color = (0, 0, 255)

        #Draw a bounding box around the voilating person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #Draw a circle around the centroid coordinates of the person
        cv2.circle(frame, (cX, cY), 5, color, 1)

    #Display the total number of social distancing voilations on the output frame
    text = "No. Of Social Distance Voilations: {}".format(len(voilate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0 ,255), 3)

    #Displaying the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    #if an output video file path has been supplied and the video writer
    #has not been initialized, do so now
    if r"social-distance-detector" != "" and writer is None:
        #Initialize the video writer 
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    #If the video writer not NONE, write the frame to the output video file
    if writer is not None:
        writer.write(frame)
cv2.destroyAllWindows()


    
    





















