import dlib
import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr
from os import listdir
from os.path import join
import time


class face_recognition(object):
    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            './models/mmod_human_face_detector.dat')

        self.face_detector = dlib.get_frontal_face_detector()

        self.pose_predictor_5_point = dlib.shape_predictor(
            './models/shape_predictor_5_face_landmarks.dat')

        self.face_encoder = dlib.face_recognition_model_v1(
            './models/dlib_face_recognition_resnet_model_v1.dat')

    def bounds(self, rect, image_shape):
        # 檢查是否超出圖片邊界
        return max(rect.top(), 0), min(rect.right(), image_shape[1]), min(rect.bottom(), image_shape[0]), max(rect.left(), 0)

    def face_detection(self, img, number_of_times_to_upsample=1, model="hog"):
        # 使用dlib的face_detecctor偵測人臉位置
        if model != "cnn":
            return [self.bounds(face, img.shape) for face in self.face_detector(img, number_of_times_to_upsample)]
        else:
            return [self.bounds(face.rect, img.shape) for face in self.cnn_face_detector(img, number_of_times_to_upsample)]

    def face_encodings(self, face_image, face_locations=None, num_jitters=1):

        # Given an image, return the 128-dimension face encoding for each face in the image.

        pose_predictor = self.pose_predictor_5_point

        if face_locations is None:
            face_locations = self.face_detection(face_image, model="cnn")
            raw_landmarks = [pose_predictor(
                face_image, self._css_to_rect(face_location)) for face_location in face_locations]
        else:
            raw_landmarks = [pose_predictor(face_image, self._css_to_rect(
                face_location)) for face_location in face_locations]

        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    def _css_to_rect(self, css):

        # Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):

       # Compare a list of face encodings against a candidate encoding to see if they match.

        if len(known_face_encodings) == 0:
            return np.empty((0))
        similars = np.linalg.norm(
            known_face_encodings - face_encoding_to_check, axis=1)
        print(similars)
        return list(similars <= tolerance)

    def compare_faces_ssim(self, known_face_encodings, face_encoding_to_check, tolerance1=0.65, tolerance2=0.4):

        similars_list = []
        num = 0
        for face in known_face_encodings:
            similars_ssim = compare_ssim(face, face_encoding_to_check)
            similars_nrmse = compare_nrmse(face, face_encoding_to_check)
            if similars_ssim >= tolerance1 and similars_nrmse <= tolerance2:
                similars_list.append((num, similars_ssim, similars_nrmse))
            else:
                print("="*5,(num, similars_ssim, similars_nrmse))
            num += 1
        print(similars_list)
        if len(similars_list) == 0:
            return 0
        else:
            return sorted(similars_list, key=lambda x: x[1], reverse=True)[0]


class people(object):
    def __init__(self, name):
        self.name = name
        self.time = 0
        self.enter = False


def load_img(fr, known_face_encodings, known_face_names, people_object_list):
    files = listdir("images")
    known_num = len(known_face_names)
    for f in files:
        img_path = join("images", f)
        img = cv2.imread(img_path)
        name = f.split(".")[0]
        if name not in known_face_names:
            face_encoding = fr.face_encodings(img)
            if len(face_encoding) == 0:
                print("encoding error")
                continue
            else:
                known_face_names.append(name)
                known_face_encodings.append(face_encoding[0])

            people_num = people("people_" + str(known_num))
            people_object_list.append(people_num)

            known_num += 1

    return known_face_encodings, known_face_names, people_object_list


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

    known_face_encodings, known_face_names, people_object_list = load_img(fr, [], [
    ], [])
    num = len(known_face_names)

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

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces_ssim(
                known_face_encodings, face_encoding)
            name = "Unknown"
            if matches != 0:
                name = known_face_names[matches[0]]
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            face_names.append(name)

        # Display the results

        for face_location, name in zip(face_detections, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            (top, right, bottom, left) = face_location

            if name == "Unknown":
                new_image = small_frame[top:bottom, left:right]

                if len(fr.face_encodings(new_image)) == 0:
                    print("encodings error")
                    continue
                cv2.imshow('un_image', new_image)
                people_num = "people_" + str(num)
                cv2.imwrite("images/" + people_num + ".jpg", new_image)
                in_window_names.append(people_num)
                num += 1
            else:
                in_window_names.append(name)
            # Draw a box around the face
            top *= int(1/zoom)
            right *= int(1/zoom)
            bottom *= int(1/zoom)
            left *= int(1/zoom)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('face_recognition', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        known_face_encodings, known_face_names, people_object_list = load_img(
            fr, known_face_encodings, known_face_names, people_object_list)

        run_time = time.time()-start
        for ele in people_object_list:
            if ele.name in in_window_names:
                ele.time += run_time
                ele.enter = True
                print(ele.name, ele.time)
        print(run_time)
    
    enter_num = 0
    for ele in people_object_list:
        if ele.enter == True:
            print(ele.name + ":",ele.time)
            enter_num += 1
    print("total people:", enter_num)
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
