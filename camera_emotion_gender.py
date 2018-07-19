from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import dlib
from imutils.face_utils import FaceAligner

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = 'detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
gender_model_path = 'gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame', cv2.WINDOW_NORMAL)
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=200)

while True:

    ret , bgr_image = video_capture.read()
    if ret == True:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    else:
        print('===')
        continue
    
    # faces = detect_faces(face_detection, gray_image)
    # for face_coordinates in faces:
    #     x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    #     rgb_face = rgb_image[y1:y2, x1:x2]

    #     x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    #     gray_face = gray_image[y1:y2, x1:x2]
    
    face_rects, scores, idx = detector.run(bgr_image, 0)
    for i, d in enumerate(face_rects):

        # text = "%2.2f(%d)" % (scores[i], idx[i])
        # x1, x2, y1, y2 = apply_offsets(d, gender_offsets)
        # rgb_face = rgb_image[y1:y2, x1:x2]

        # x1, x2, y1, y2 = apply_offsets(d, emotion_offsets)
        # gray_face = gray_image[y1:y2, x1:x2]

        rgb_face_fa = fa.align(rgb_image, rgb_image, d)
        gray_face_fa = fa.align(gray_image, gray_image, d)
        # cv2.imshow('window_frame1', rgb_face_fa)
        # cv2.imshow('window_frame2', gray_face_fa)
        try:
            rgb_face_fa = cv2.resize(rgb_face_fa, (gender_target_size))
            gray_face_fa = cv2.resize(gray_face_fa, (emotion_target_size))
        except:
            continue
        
        rgb_face_fa = preprocess_input(rgb_face_fa, False)
        rgb_face_fa = np.expand_dims(rgb_face_fa, 0)
        gender_prediction = gender_classifier.predict(rgb_face_fa)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        print(gender_text)

        gray_face_fa = preprocess_input(gray_face_fa, True)
        gray_face_fa = np.expand_dims(gray_face_fa, 0)
        gray_face_fa = np.expand_dims(gray_face_fa, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face_fa))
        emotion_text = emotion_labels[emotion_label_arg]
        print(emotion_text)

        # gray_face = preprocess_input(gray_face, True)
        # gray_face = np.expand_dims(gray_face, 0)
        # gray_face = np.expand_dims(gray_face, -1)
        # emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        # emotion_text = emotion_labels[emotion_label_arg]
        # emotion_window.append(emotion_text)

        # rgb_face = np.expand_dims(rgb_face, 0)
        # rgb_face = preprocess_input(rgb_face, False)
        # gender_prediction = gender_classifier.predict(rgb_face)
        # gender_label_arg = np.argmax(gender_prediction)
        # gender_text = gender_labels[gender_label_arg]
        # gender_window.append(gender_text)
        # if len(gender_window) > frame_window:
        #     emotion_window.pop(0)
        #     gender_window.pop(0)
        # try:
        #     emotion_mode = mode(emotion_window)
        #     gender_mode = mode(gender_window)
        # except:
        #     continue

        if gender_text == gender_labels[0]:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        draw_bounding_box(d, rgb_image, color)
        draw_text(d, rgb_image, gender_text, color, 0, -20, 1, 1)
        draw_text(d, rgb_image, emotion_text, color, 0, -45, 1, 1)

        # draw_bounding_box(face_coordinates, rgb_image, color)
        # draw_text(face_coordinates, rgb_image, gender_mode,
        #           color, 0, -20, 1, 1)
        # draw_text(face_coordinates, rgb_image, emotion_mode,
        #           color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
