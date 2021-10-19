import yolo3_one_file_to_detect_them_all as yolo3
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2
#===========================================================
# specify data file name to run object detection on
file_name = 'data/diningroom.jpg'

# specify threshold for decoding bbox output
obj_thresh = 0.5
nms_thresh = 0.6
#===========================================================
# yolo3 parameters
net_h, net_w = 416, 416
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Define the Model
model = yolo3.make_yolov3_model()

# load the model weights from file yolov3.weights
weight_reader = yolo3.WeightReader('yolov3.weights')

# set the model weights into the model
weight_reader.load_weights(model)

# # save the model for later use
# model.save('yolov3.h5')

#===========================================================
# load an image
image = cv2.imread(file_name)

# keep image width and height
image_h, image_w, _ = image.shape

# YOLOv3 model expects input shape = (416, 416, 3)
image = cv2.resize(image, (net_h, net_w))


# preprocess input
# option 1 : reshape and normalize image manually
# option 2 : preprocess_input of yolo3
image = yolo3.preprocess_input(image, net_h, net_w)

#===========================================================
# Make prediction
yhat = model.predict(image)

# decode the bounding boxes from the yhat
boxes = list()
for i in range(len(yhat)):
    boxes += yolo3.decode_netout(yhat[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
    
# correct the sizes of the bounding boxes
yolo3.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

# suppress non-maximal boxes 
# to merge the overlap boxes to the same object
yolo3.do_nms(boxes, nms_thresh)  

#===========================================================
# get details of the valid objects
v_boxes, v_labels, v_scores = list(), list(), list()

for box in boxes :
    for i in range(len(labels)):
        if box.classes[i] > obj_thresh:
            v_boxes.append(box)
            v_labels.append(labels[i])
            v_scores.append(box.classes[i]*100)
            
# summarize what we found
for i in range(len(v_boxes)):
    print('%s:%.3f' %(v_labels[i], v_scores[i]))

# load image as numpy array
image = plt.imread(file_name)

# plot each box
for box in v_boxes :
    # get coordinates
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    
    # create rectangle
    cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,255), 2)
    
    # add text and score
    cv2.putText(image, 
                f"{v_labels[i]} {round(v_scores[i],3)}%", 
                (x1,y1-5), 
                cv2.FONT_HERSHEY_PLAIN, 1, 
                (255,255,255), 2)

# save the result
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite(file_name.split('.')[0]+'_detection.jpg', (image).astype('uint8')) 

# show the detection result using cv2
cv2.imshow('detection (Press any key to quit)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------------------
# # option 2 : draw bounding boxes on the image using yolo3.draw_boxes function
# image = plt.imread(file_name)
# yolo3.draw_boxes(image, boxes, labels, obj_thresh) 

# # save the result
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imwrite(file_name.split('.')[0]+'_detection.jpg', (image).astype('uint8')) 

# # show the detection result using cv2
# cv2.imshow('detection (Press any key to quit)', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
