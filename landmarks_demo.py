#人臉特徵點demo

import cv2
import dlib
from os import listdir
from os.path import join
import sys


pose_predictor = dlib.shape_predictor(
    "./models/shape_predictor_5_face_landmarks.dat")

# 特徵點數量
num = 5

face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    './models/mmod_human_face_detector.dat')

width = 1280
height = 720
zoom = 1
if len(sys.argv) == 1:
    print("select mode (WIN or TX2)")
    sys.exit()
elif sys.argv[1] == 'TX2':
    # 在TX2上使用視訊鏡頭
    gst_str = ("nvcamerasrc ! "
               "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(width, height)

    video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    model = "cnn"
elif sys.argv[1] == 'WIN':
        # 在windows使用視訊頭
    video_capture = cv2.VideoCapture(0)
    model = "hog"
else:
    print("select mode (WIN or TX2)")
    sys.exit()

# with open (path) as file:
while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if model == "cnn":
        faces = cnn_face_detector(rgb_frame, 1)
    else:
        faces = face_detector(rgb_frame, 1)
    # print(faces)
    for face in faces:
        if model == "cnn":
            location = (face.rect.top(), face.rect.right(),
                        face.rect.bottom(), face.rect.left())
            sorce = face.confidence
            print(sorce)
            roi_gray = small_frame[face.rect.top():face.rect.bottom(
            ), face.rect.left():face.rect.right()]
        else:
            location = (face.top(), face.right(), face.bottom(), face.left())

            roi_gray = small_frame[face.top():face.bottom(),
                                   face.left():face.right()]

        raw_landmark = pose_predictor(rgb_frame, dlib.rectangle(
            location[3], location[0], location[1], location[2]))

        for i in range(num):
            print(raw_landmark.part(i))
            cv2.circle(small_frame, (raw_landmark.part(i).x,
                                     raw_landmark.part(i).y), 5, (0, 0, 255), -1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('landmarks', small_frame)
video_capture.release()
cv2.destroyAllWindows()
