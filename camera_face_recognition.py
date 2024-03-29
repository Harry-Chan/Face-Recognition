
import face_recognition as FR
import emotion_gender as EG
import cv2
from keras import backend
from os import listdir
from os.path import join
import math
import time
import sys


class people(object):
    def __init__(self, name, face_encoding):
        self.name = name
        self.face_encoding = face_encoding  # 人臉編碼
        self.time = 0  # 在畫面停留時間
        self.center = (0, 0)  # 人臉在畫面的中心位置
        self.enter = False  # 是否有進入到畫面過

    def cal_center(self, face_location):  # 計算人臉中心點
        (top, right, bottom, left) = face_location
        self.center = ((left+right) / 2, (bottom+top) / 2)


# 讀取已知人臉
def load_img(fr, known_face_names, people_objects):
    files = listdir("images")
    known_num = len(known_face_names)  # 已知人臉數量
    if len(files) == known_num:  # 判斷是否有新的人臉加入到file中，沒有則直接回傳
        return people_objects, known_face_names, known_num
    else:
        for f in files:
            img_path = join("images", f)
            name = f.split(".")[0]
            if name not in known_face_names:  # 找出新的人臉，產生對應的people object
                img = cv2.imread(img_path)
                face_encoding = fr.face_encodings(img)
                if len(face_encoding) == 0:  # 如果人臉編碼有錯則不新增
                    print("encoding error")
                    continue
                else:
                    new_name = "people_" + str(known_num)
                    new_people = people(new_name, face_encoding)
                    people_objects.append(new_people)
                    known_face_names.append(new_name)
                    known_num += 1

        return people_objects, known_face_names, known_num


def main():

    width = 1280
    height = 720
    fps = 120
    zoom = 0.5
    if len(sys.argv) == 1:
        print("select mode (WIN or TX2)")
        sys.exit()
    elif sys.argv[1] == 'TX2':
        # 在TX2上使用視訊鏡頭
        # 有三個模式可選擇
        # 2592 * 1944 ,fps 30
        # 2592 * 1458 ,fps 30
        # 1280 * 720 ,fps 120
        gst_str = ("nvcamerasrc ! "
                   "video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)I420, framerate=(fraction){}/1 ! "
                   "nvvidconv ! video/x-raw, format=(string)BGRx ! "
                   "videoconvert ! appsink").format(width, height, fps)

        video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        model = "cnn"  # 人臉偵測的model選擇
    elif sys.argv[1] == 'WIN':
        # 在windows使用視訊頭
        video_capture = cv2.VideoCapture(0)
        model = "hog"
    else:
        print("select mode (WIN or TX2)")
        sys.exit()

    # 初始化套件
    fr = FR.face_recognition()  # 人臉辨識
    eg = EG.emotion_gender()  # 表情、性別預測

    # 讀取已知人臉圖片
    people_objects, known_face_names, known_num = load_img(fr, [], [])

    while True:
        start = time.time()
        # 目前在畫面上的人名
        in_window_names = []

        # 讀取視訊畫面
        ret, frame = video_capture.read()

        # Resize 大小加速運算速度
        small_frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)

        # 將BRG(openCV使用)轉成RGB模式
        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 偵測畫面中的人臉位置(可使用cnn與hog模式)
        face_detections = fr.face_detection(small_frame, model=model)

        # 取出人臉特徵點並轉換成128維的特徵向量
        face_encodings, faces_images = fr.face_encodings(
            small_frame, face_detections)

        # 將人臉依照眼睛位置對齊(轉正)
        # image_aligners = fr.face_aligners(small_frame, face_detections)

        # 與原先已知的人臉比對，查看是否已存在
        for face_location, face_encoding, faces_image in zip(face_detections, face_encodings, faces_images):

            # 向量為空list，代表抓到側臉
            if len(face_encoding) == 0:
                print("face error")
                continue
            # 進行比對
            matches = fr.compare_faces(
                face_encoding, face_location, people_objects)
        # 人臉已存在 計算位置的中心點 name新增到list中
            if matches != 0:
                name = known_face_names[matches[0]]
                people_num = people_objects[known_face_names.index(name)]
                people_num.cal_center(face_location)
                in_window_names.append(name)
        # 人臉未存在 將人臉儲存 並新增到people_objects
            else:
                name = "people_" + str(known_num)

                cv2.imwrite(
                    "images/{0}.jpg".format(name), faces_image)
                cv2.imshow('un_image', faces_image)
                new_people = people(name, face_encoding)
                new_people.cal_center(face_location)
                people_objects.append(new_people)

                known_face_names.append(name)
                known_num += 1
                in_window_names.append(name)

            cv2.imshow('found_face', faces_image)

            # 性別預測
            gender_text = eg.gender_prediction(faces_image)
            #gender_text = "man"
            # 表情預測
            emotion_text = eg.emotion_prediction(faces_image)
            #emotion_text = "happy"

        # 框出人臉與畫上label
            top, right, bottom, left = face_location
            # 將邊界放大回原本size
            top *= int(1/zoom)
            right *= int(1/zoom)
            bottom *= int(1/zoom)
            left *= int(1/zoom)

            # 設定邊界顏色
            if gender_text == 'man':
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            # 框出人臉
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), color, cv2.FILLED)

            # 將資訊寫入到畫面
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

            cv2.putText(frame, gender_text, (left, top - 6),
                        font, 1.0, (255, 255, 255), 1)

            cv2.putText(frame, emotion_text, (left, top + 20),
                        font, 1.0, (255, 255, 255), 1)

        # 顯示畫面
        cv2.imshow('face_recognition', frame)

        # 計算一個process時間
        run_time = time.time() - start

        # 將目前有出現在畫面的人臉，加上此次process時間
        for ele in people_objects:
            if ele.name in in_window_names:
                ele.time += run_time
                ele.enter = True
                print(ele.name, ele.time)
        print(run_time)

        # 按q鍵離開程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 顯示所有出現過人臉的時間資訊
    enter_num = 0
    for ele in people_objects:
        if ele.enter == True:
            print(ele.name + ":", ele.time)
            enter_num += 1
    print("total people:", enter_num)

    # Release handle to the webcam
    backend.clear_session()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
