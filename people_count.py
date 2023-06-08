import cv2
import numpy as np
import tensorflow as tf

# Load the COCO class labels for YOLOv3
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load the YOLOv3 model with TensorFlow
model = tf.keras.models.load_model("yolov3.h5")

# Load the video
cap = cv2.VideoCapture(0)

# Initialize a counter for the number of people detected
people_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for YOLOv3
    img = cv2.resize(frame, (416, 416))
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    # Predict the objects in the frame using YOLOv3
    outputs = model.predict(img)
    boxes, scores, classes, nums = outputs

    # Iterate over the detected objects
    for i in range(nums[0]):
        # If the detected object is a person, increment the counter
        if class_names[int(classes[0][i])] == "person":
            people_count += 1

        # Draw a bounding box around the detected object
        box = boxes[0][i]
        score = scores[0][i]
        label = class_names[int(classes[0][i])]
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * 416), int(y1 * 416), int(x2 * 416), int(y2 * 416)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the video with the person count and bounding boxes
    cv2.putText(frame, f"People Count: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Person Counting", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
