import dlib
import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr
from keras.models import load_model
from keras import backend
import numpy as np
from os import listdir
from os.path import join
import math
import time


class face_recognition(object):
    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            './models/mmod_human_face_detector.dat')

        #self.face_detector = dlib.get_frontal_face_detector()

        self.pose_predictor_5_point = dlib.shape_predictor(
            './models/shape_predictor_5_face_landmarks.dat')

        self.face_encoder = dlib.face_recognition_model_v1(
            './models/dlib_face_recognition_resnet_model_v1.dat')

        self.gender_classifier = load_model(
            'gender_models/simple_CNN.81-0.96.hdf5', compile=False)

        self.gender_labels = {0: 'woman', 1: 'man'}

        self.emotion_classifier = load_model(
            'emotion_models/simple_CNN.985-0.66.hdf5', compile=False)

        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                               4: 'sad', 5: 'surprise', 6: 'neutral'}

    def bounds(self, rect, image_shape):
        # 檢查是否超出圖片邊界
        if rect.top() < 0 or rect.right() > image_shape[1] or rect.bottom() > image_shape[0] or rect.left() < 0:
            return False
        else:
            return True

    def face_detection(self, img, number_of_times_to_upsample=1, model="hog"):
        # 使用dlib的face_detecctor偵測人臉位置
        if model != "cnn":
            return [(face.top(), face.right(), face.bottom(), face.left()) for face in self.face_detector(img, number_of_times_to_upsample) if self.bounds(face, img.shape) == True]
        else:
            return [(face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()) for face in self.cnn_face_detector(img, number_of_times_to_upsample) if self.bounds(face.rect, img.shape) == True]

    def face_encodings(self, face_image, face_locations=None, num_jitters=0):
        # 將人臉編碼成128維的向量
        # Given an image, return the 128-dimension face encoding for each face in the image.

        pose_predictor = self.pose_predictor_5_point

        if face_locations is None:
            bottom, right, _ = face_image.shape
            face_location = (0, right, bottom, 0)
            raw_landmarks = [pose_predictor(
                face_image, self._css_to_rect(face_location))]

            # print("---"*5, raw_landmarks[0].part(0),
            #       raw_landmarks[0].part(1), raw_landmarks[0].part(2), raw_landmarks[0].part(3), raw_landmarks[0].part(4))
        else:
            raw_landmarks = [pose_predictor(face_image, self._css_to_rect(
                face_location)) for face_location in face_locations]

        return [np.array(self.face_encoder.compute_face_descriptor(
            face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    def _css_to_rect(self, css):
        # 將像素位置轉換成dlib使用的格式
        # Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        return dlib.rectangle(css[3], css[0], css[1], css[2])

    # def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):

    #    # Compare a list of face encodings against a candidate encoding to see if they match.

    #     if len(known_face_encodings) == 0:
    #         return np.empty((0))
    #     similars = np.linalg.norm(
    #         known_face_encodings - face_encoding_to_check, axis=1)
    #     print(similars)
    #     return list(similars <= tolerance)

    def compare_faces_ssim(self, face_encoding_to_check, face_location_to_check, people_object_list, tolerance1=0.7, tolerance2=0.4):
        # 比較人臉相似度
        similars_list = []
        num = 0
        (top, right, bottom, left) = face_location_to_check
        x, y = ((left+right) / 2, (bottom+top) / 2)

        for people in people_object_list:
            # 使用ssim(結構相似性)
            similars_ssim = compare_ssim(
                people.face_encoding, face_encoding_to_check)
            # 使用nrmse(正規化方均根差)
            similars_nrmse = compare_nrmse(
                people.face_encoding, face_encoding_to_check)
            # 計算兩張圖的中心距離
            center_distance = (
                (x - people.center[0]) ** 2 + (y - people.center[1]) ** 2) ** 0.5
            print("center_distance", center_distance)
            if similars_ssim >= tolerance1 and similars_nrmse <= tolerance2:
                similars_list.append((num, similars_ssim, similars_nrmse))
            elif center_distance <= 50 and (similars_ssim >= tolerance1 - 0.2 or similars_nrmse <= tolerance2 + 0.2):
                similars_list.append((num, similars_ssim, similars_nrmse))
                print("============")
            else:
                print("="*5, (num, similars_ssim, similars_nrmse))
            num += 1
        print(similars_list)
        if len(similars_list) == 0:
            return 0
        else:
            return sorted(similars_list, key=lambda x: x[1], reverse=True)[0]

    def gender_prediction(self, image):
        # rgb_face_fa = preprocess_input(rgb_face_fa, False)
        gender_input_size = self.gender_classifier.input_shape[1:3]
        image = cv2.resize(image, (gender_input_size))
        image = image / 255
        image = np.expand_dims(image, 0)
        gender_prediction = self.gender_classifier.predict(image)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = self.gender_labels[gender_label_arg]
        print('gender', gender_text)
        return gender_text

    def emotion_prediction(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        emotion_input_size = self.emotion_classifier.input_shape[1:3]
        gray_image = cv2.resize(gray_image, (emotion_input_size))
        gray_image = gray_image/255
        gray_image = np.expand_dims(gray_image, 0)
        gray_image = np.expand_dims(gray_image, -1)

        emotion_prediction = self.emotion_classifier.predict(gray_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.emotion_labels[emotion_label_arg]
        print('emotion', emotion_text)
        return emotion_text


class people(object):
    def __init__(self, name, face_encoding):
        self.name = name
        self.face_encoding = face_encoding
        self.time = 0
        self.center = (0, 0)
        self.enter = False

    def cal_center(self, face_location):
        (top, right, bottom, left) = face_location
        self.center = ((left+right) / 2, (bottom+top) / 2)


def load_img(fr, known_face_names, people_object_list):
    files = listdir("images")
    known_num = len(known_face_names)
    if len(files) == known_num:
        return people_object_list, known_face_names, known_num
    else:
        for f in files:
            img_path = join("images", f)
            name = f.split(".")[0]
            if name not in known_face_names:
                img = cv2.imread(img_path)
                face_encoding = fr.face_encodings(img)
                if len(face_encoding) == 0:
                    print("encoding error")
                    continue
                else:
                    new_name = "people_" + str(known_num)
                    new_people = people(new_name, face_encoding[0])
                    people_object_list.append(new_people)

                    known_face_names.append(new_name)
                    known_num += 1

        return people_object_list, known_face_names, known_num


def main():

    width = 1280
    height = 720
    zoom = 0.5
    gst_str = ("nvcamerasrc ! "
               "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1944, format=(string)I420, framerate=(fraction)30/1 ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(width, height)

    video_capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    #video_capture = cv2.VideoCapture(0)
    fr = face_recognition()

    people_object_list, known_face_names, known_num = load_img(fr, [], [])

    while True:
        start = time.time()
        in_window_names = []
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_detections = fr.face_detection(rgb_small_frame, model="cnn")

        face_encodings = fr.face_encodings(rgb_small_frame, face_detections)

        face_names = []

        for face_location, face_encoding in zip(face_detections, face_encodings):
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces_ssim(
                face_encoding, face_location, people_object_list)
            name = "Unknown"
            if matches != 0:
                name = known_face_names[matches[0]]
            face_names.append(name)

        # Display the results

        for face_location, name in zip(face_detections, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top, right, bottom, left = face_location
            image = small_frame[top:bottom, left:right]
            if name == "Unknown":
                face_encoding = fr.face_encodings(image)
                if len(face_encoding) == 0:
                    print("encodings error")
                    continue
                else:
                    new_name = "people_" + str(known_num)
                    cv2.imshow('un_image', image)
                    cv2.imwrite(
                        "images/{0}.jpg".format(new_name), image)

                    new_people = people(new_name, face_encoding[0])
                    new_people.cal_center(face_location)
                    people_object_list.append(new_people)

                    known_face_names.append(new_name)
                    known_num += 1
                    in_window_names.append(new_name)

            else:
                people_num = people_object_list[known_face_names.index(name)]
                people_num.cal_center(face_location)
                in_window_names.append(name)

            gender_text = fr.gender_prediction(image)
            emotion_text = fr.emotion_prediction(image)
            # Draw a box around the face
            top *= int(1/zoom)
            right *= int(1/zoom)
            bottom *= int(1/zoom)
            left *= int(1/zoom)

            # Draw a label with a name below the face
            if gender_text == 'man':
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (255, 0, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (255, 0, 0), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

            cv2.putText(frame, gender_text, (left, top - 6),
                        font, 1.0, (255, 255, 255), 1)

            cv2.putText(frame, emotion_text, (left, top + 20),
                        font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        cv2.imshow('face_recognition', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        run_time = time.time() - start
        for ele in people_object_list:
            if ele.name in in_window_names:
                ele.time += run_time
                ele.enter = True
                print(ele.name, ele.time)
        print(run_time)

    enter_num = 0
    for ele in people_object_list:
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
