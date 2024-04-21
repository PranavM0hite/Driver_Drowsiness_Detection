import cv2
import dlib
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import time
import pygame
import numpy as np
import streamlit as st

# Initialize pygame
pygame.mixer.init()

# Load the drowsiness detection model
model = load_model('drowsiness_detection_model.h5')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    mar = (A + B + C) / 3.0
    return mar

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 40
LOOK_FORWARD_CONSEC_FRAMES = 30
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_CONSEC_FRAMES = 25

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

vs = VideoStream(src=0).start()
time_face_not_detected = None
elapsed_time_frames = 0
COUNTER_EYE = 0
COUNTER_MOUTH = 0

# Load the alarm sound
pygame.mixer.music.load("data/alarm.wav")

st.markdown('<h1 style="text-align: center;">Jaagte Rahoo !!!</h1>', unsafe_allow_html=True)

frame_placeholder = st.empty()

while True:
    frame = vs.read()
    if frame is None:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    if len(rects) > 0:
        time_face_not_detected = None
        elapsed_time_frames = 0  # Reset the time when face is detected
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            mouthMAR = mouth_aspect_ratio(mouth)

            ear = (leftEAR + rightEAR) / 2.0

            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Drowsiness detection using CNN model
            eye_region = gray[rect.top():rect.bottom(), rect.left():rect.right()]
            eye = cv2.resize(eye_region, (145, 145))
            eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
            eye = eye.astype('float') / 255.0
            eye = img_to_array(eye)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)
            if np.argmax(prediction) == 1:
                pygame.mixer.music.play()

            if ear < EYE_AR_THRESH:
                COUNTER_EYE += 1
                if COUNTER_EYE == EYE_AR_CONSEC_FRAMES:
                    pygame.mixer.music.play()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                COUNTER_EYE = 0

            if mouthMAR > MOUTH_AR_THRESH:
                COUNTER_MOUTH += 1
                if COUNTER_MOUTH == MOUTH_AR_CONSEC_FRAMES:
                    pygame.mixer.music.play()
                    cv2.putText(frame, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            else:
                COUNTER_MOUTH = 0

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mouthMAR:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        if time_face_not_detected is None:
            time_face_not_detected = time.time()
        else:
            elapsed_time_frames += 1
            for _ in range(LOOK_FORWARD_CONSEC_FRAMES):
                cv2.putText(frame, "LOOK FORWARD ALERT!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if elapsed_time_frames == LOOK_FORWARD_CONSEC_FRAMES:
                    pygame.mixer.music.play()   
                    break

    frame_placeholder.image(frame, channels="BGR")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
