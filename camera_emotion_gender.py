import cv2
from keras.models import load_model
import numpy as np
import dlib
from imutils.face_utils import FaceAligner

from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.preprocessor import preprocess_input

# parameters for loading data and images
emotion_model_path = 'emotion_models/fer2013_mini_XCEPTION.107-0.66.hdf5'
gender_model_path = 'gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

# 開啟影片檔案
width = 640
height = 360
gst_str = ("nvcamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
            "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
            "videoconvert ! appsink").format(width, height)

video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

#video_capture = cv2.VideoCapture()  #on windows

# starting video streaming
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=200)

while(video_capture.isOpened()):

    ret , bgr_image = video_capture.read()
    if ret == True:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    else:
        continue

    face_rects, scores, idx = detector.run(gray_image, 0)
    for i, d in enumerate(face_rects):

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


        if gender_text == gender_labels[0]:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        draw_bounding_box(d, rgb_image, color)
        draw_text(d, rgb_image, gender_text, color, 0, -20, 1, 1)
        draw_text(d, rgb_image, emotion_text, color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Emotion_Recognition', bgr_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
