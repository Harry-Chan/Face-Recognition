import cv2
import dlib
from os import listdir
from os.path import join

pose_predictor = dlib.shape_predictor("predictor.dat")

face_detector = dlib.get_frontal_face_detector()
path = 'annotation/332.txt'
video_capture = cv2.VideoCapture(0)

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame,0)
    print(faces)
    for face in faces:
        location = (face.top(), face.right(), face.bottom(), face.left())
        raw_landmark = pose_predictor(rgb_frame, dlib.rectangle(location[3], location[0], location[1], location[2]))
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
            print(raw_landmark.part(i))
            cv2.circle(frame, (raw_landmark.part(i).x,raw_landmark.part(i).y), 5, (0, 0, 255), -1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('test2', frame)
video_capture.release()
cv2.destroyAllWindows()    
    # cv2.imshow('test', Oimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()