import cv2
import numpy as np
# Load image
image = cv2.imread('example.jpg')
# Load pre-trained model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
# Load class labels
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Set input size and scale factor
input_size = (416, 416)
scale = 1/255.0

# Preprocess image
blob = cv2.dnn.blobFromImage(image, scale, input_size, swapRB=True, crop=False)

# Set input to the network
net.setInput(blob)

# Forward pass and get output
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

# Process detections
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * 300)
            center_y = int(detection[1] * 200)
            w = int(detection[2] * 200)
            h = int(detection[3] * 300)

            # Draw bounding box
            cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
            cv2.putText(image, classes[class_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display result
cv2.imshow('Localization Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()