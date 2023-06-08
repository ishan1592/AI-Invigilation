import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        outer_mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 61)])
        inner_mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(61, 68)])

        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        #mouth_width=43
        top_lip = np.mean(inner_mouth[3:5], axis=0)
        bottom_lip = np.mean(inner_mouth[6:8], axis=0)
        mouth_height = np.linalg.norm(bottom_lip - top_lip)

        print(mouth_height)
        print('---')
        print(mouth_width)
        print('===')
        mar = mouth_height / mouth_width
        #print(mar)
        if mar > 0.42:
            cv2.putText(frame, "Mouth Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for i in range(48, 61):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
        for i in range(61, 68):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)

    cv2.imshow("Mouth Opening Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
