import sys
sys.path.append('/content/darknet')
import sys

import cv2
import darknet
import matplotlib.pyplot as plt

config_file_path = '/content/darknet/cfg/yolov4.cfg'
data_file_path = '/content/darknet/cfg/coco.data'
weights_file_path = '/content/darknet/yolov4.weights'


network, class_names, class_colors = darknet.load_network(config_file_path,
                                                  data_file_path, 
                                                  weights_file_path,
                                                  batch_size=1)
#Darknet doesn't accept numpy images.
#Create one with image we reuse for each detect
net_width = darknet.network_width(network)
net_height = darknet.network_height(network)


img = cv2.imread('/content/darknet/data/people.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(img, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

darknet_image = darknet.make_image(net_width, net_height, 3)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

conf_threshold = 0.4
iou_threshold = 0.3
detections = darknet.detect_image(network, 
                                class_names, 
                                darknet_image,  
                                thresh= conf_threshold, 
                                hier_thresh=.5, 
                                nms= iou_threshold)

print(detections)
print(detections)

def plot_my_image(img, results):
    for class_name, confidence, box in results:
        if class_name != 'person':
            continue
        else:
            xmin, ymin, xmax, ymax = darknet.bbox2points(box) 

            # Start coordinate, here (5, 5)
            # represents the top left corner of rectangle
            start_point = (xmin, ymin)
    
            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            end_point = (xmax, ymax)
    
            # Blue color in BGR
            color = (255, 0, 0)
    
            # Line thickness of 2 px
            thickness = 2
    
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img

output = plot_my_image(frame_resized, detections) 

plt.figure(figsize=(10,10))
plt.imshow(output)
plt.axis('off')
plt.savefig('detection-result-plot2.png', dpi=80, bbox_inches='tight')

