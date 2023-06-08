import cv2
import dlib
import numpy as np

# Load face detection and facial landmark detection models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load face recognition model and reference image
facerec = dlib.face_recognition_model_v1('/home/ishan/PycharmProjects/pythonProject/dlib_face_recognition_resnet_model_v1.dat')
img = cv2.imread("reference_image1.jpg")
dets = detector.run(img, 1)
for i, d in enumerate(dets):
    print("Detection {}, score: {}, face_type:{}".format(
        d, scores[i], idx[i]))
    crop = img[d.top():d.bottom(), d.left():d.right()]
    cv2.imwrite("cropped.jpg", crop)
reference_image = cv2.imread('/home/ishan/PycharmProjects/pythonProject/cropped.jpg')
reference_encoding = facerec.compute_face_descriptor(reference_image)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through detected faces
    for face in faces:
        # Detect facial landmarks in the face region
        landmarks = predictor(gray, face)

        # Convert landmarks to NumPy array for convenience
        landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # Compute face descriptor using the facial landmarks and face recognition model
        face_encoding = facerec.compute_face_descriptor(frame, landmarks)

        # Compare face encoding with reference encoding
        distance = np.linalg.norm(np.array(face_encoding) - np.array(reference_encoding))
        print(distance)
        # If distance between encodings is small, it's the same person
        if distance < 0.6:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Person Detected", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Otherwise, it's a different person
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "Proxy Detected", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
