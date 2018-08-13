import cv2
import dlib
from os import listdir
from os.path import join
import sys


pose_predictor = dlib.shape_predictor("./models/predictor_68_new.dat")

face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    './models/mmod_human_face_detector.dat')
width = 1280
height = 720
zoom = 0.5
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

    num = 0
    num2 = 0
    # for line in file:
    #     line = line.strip()
    #     if num == 0:
    #         image_name = line + '.jpg'
    #         print(image_name)
    #         img_path = join("train_1", image_name)
    #         print(img_path)
    #         img = cv2.imread(img_path)
    #         Oimg = cv2.imread(img_path)
    #         cv2.imshow('test2', img)
    # face = face_detector(img,0)
    small_frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = cnn_face_detector(rgb_frame, 1)
    #faces = face_detector(rgb_frame,1)
    # print(faces)
    for face in faces:
        location = (face.rect.top(), face.rect.right(),
                    face.rect.bottom(), face.rect.left())
        sorce = face.confidence
        print(sorce)
     #   location = (face.top(), face.right(), face.bottom(), face.left())
        raw_landmark = pose_predictor(rgb_frame, dlib.rectangle(
            location[3], location[0], location[1], location[2]))
        # break

        # top = face[0].top()
        # left = face[0].left()
        # width = face[0].right() - left
        # height = face[0].bottom() - top

    #         print(top,left,width,height)

    #     elif num % 3 == 1 or num == 124 or num == 145 or num == 50 :
    #         num2 += 1
    #         tmp = line.split(' , ')
    #         cv2.circle(img, (round(float(tmp[0])),round(float(tmp[1]))), 5, (0, 0, 255), -1)
    #     cv2.imshow('test2', img)
    #     cv2.waitKey(0)
    #     print(num)
    #     num += 1
    # print("2",num2)
        for i in range(68):
            # print(raw_landmark.part(i))
            cv2.circle(small_frame, (raw_landmark.part(i).x,
                                     raw_landmark.part(i).y), 5, (0, 0, 255), -1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('test2', small_frame)
video_capture.release()
cv2.destroyAllWindows()
# cv2.imshow('test', Oimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
