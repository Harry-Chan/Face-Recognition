import dlib
import numpy as np
import cv2
from face_aligner import FaceAligner
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr


class face_recognition(object):
    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            './models/mmod_human_face_detector.dat')

        self.face_detector = dlib.get_frontal_face_detector()

        self.pose_predictor_5_point = dlib.shape_predictor(
            './models/shape_predictor_5_face_landmarks.dat')

        #self.pose_predictor_5_point = dlib.shape_predictor(
        #    './models/shape_predictor_68_face_landmarks.dat')

        self.face_encoder = dlib.face_recognition_model_v1(
            './models/dlib_face_recognition_resnet_model_v1.dat')

        self.face_aligner = FaceAligner(
            self.pose_predictor_5_point, desiredFaceWidth=500)

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
            # hyper-parameters for bounding boxes shape
            x, y = (20, 20)
            return [(face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()) for face in self.cnn_face_detector(img, number_of_times_to_upsample) if self.bounds(face.rect, img.shape) == True and face.confidence > 1]

    def face_encodings(self, image, face_locations=None, num_jitters=1):
        # 將人臉編碼成128維的向量
        # Given an image, return the 128-dimension face encoding for each face in the image.
        pose_predictor = self.pose_predictor_5_point

        if face_locations is None:
            cv2.imshow('known_face', image)
            bottom, right, _ = image.shape
            face_location = (0, right, bottom, 0)
            raw_landmark = pose_predictor(
                image, self._css_to_rect(face_location))

            return np.array(self.face_encoder.compute_face_descriptor(image, raw_landmark, num_jitters))
            # print("---"*5, raw_landmarks[0].part(0),
            #       raw_landmarks[0].part(1), raw_landmarks[0].part(2), raw_landmarks[0].part(3), raw_landmarks[0].part(4))
        else:
            raw_landmarks = []
            faces_images = []
            face_encodings = []
            for face_location in face_locations:
                top, right, bottom, left = face_location
                faces_image = image[top:bottom, left:right]
                faces_image = cv2.resize(faces_image, (200, 200))
                faces_images.append(faces_image)
                raw_landmark = pose_predictor(
                    faces_image, self._css_to_rect((0, 200, 200, 0)))
                raw_landmarks.append(raw_landmark)

                test = cv2.resize(image[top:bottom, left:right], (200, 200))
                #for i in range(5):
                #    cv2.circle(test, (raw_landmark.part(i).x,raw_landmark.part(i).y), 5, (0, 0, 255), -1)
                #cv2.imshow('test', test)
                
                if raw_landmark.part(4).x > 145 or raw_landmark.part(4).x < 55 or raw_landmark.part(4).y < 105:
                    print("nose", raw_landmark.part(
                        4).x, raw_landmark.part(4).y)
                    face_encodings.append([])
                    continue

                face_encoding = np.array(self.face_encoder.compute_face_descriptor(
                    faces_image, raw_landmark, num_jitters))
                face_encodings.append(face_encoding)
            # face_encodings = [np.array(self.face_encoder.compute_face_descriptor(
            #    image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
            return face_encodings, faces_images

    def face_aligners(self, face_image, face_locations):
        face_aligners = []

        for face_location in face_locations:

            face_aligner = self.face_aligner.align(
                face_image, face_image, self._css_to_rect(face_location))
            cv2.imshow('found_face', face_aligner)
            face_aligners.append(face_aligner)

        return face_aligners

    def _css_to_rect(self, css):
        # 將像素位置轉換成dlib使用的格式
        # Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def compare_faces(self, face_encoding_to_check, face_location_to_check, people_object_list, tolerance1=0.7, tolerance2=0.6):
        # 比較人臉相似度
        similars_list = []

        (top, right, bottom, left) = face_location_to_check
        x, y = ((left+right) / 2, (bottom+top) / 2)

        num = 0
        for people in people_object_list:
            # 使用ssim(結構相似性)
            similars_ssim = compare_ssim(
                people.face_encoding, face_encoding_to_check)

            # 使用nrmse(正規化方均根差)
            # similars_nrmse = compare_nrmse(
            #     people.face_encoding, face_encoding_to_check)

            # 計算向量的歐式距離
            encoding_distance = np.linalg.norm(
                people.face_encoding - face_encoding_to_check)
            # 計算兩張圖的中心距離
            center_distance = (
                (x - people.center[0]) ** 2 + (y - people.center[1]) ** 2) ** 0.5
            print("center_distance", center_distance)

            if similars_ssim >= tolerance1 and encoding_distance <= tolerance2:
                # Harmonic Mean
                HM = self.Harmonic_Mean(similars_ssim, encoding_distance)
                similars_list.append((num, HM))
                print("OO"*5, (num, similars_ssim, encoding_distance))
            elif center_distance <= 30 and (similars_ssim >= tolerance1 or encoding_distance <= tolerance2):
                # Harmonic Mean
                HM = self.Harmonic_Mean(similars_ssim, encoding_distance)

                similars_list.append((num, HM))
                print("center_distance <= 30")
                print("OO"*5, (num, similars_ssim, encoding_distance))
            else:
                print("XX"*5, (num, similars_ssim, encoding_distance))
            num += 1
        print(similars_list)
        if len(similars_list) == 0:
            return 0
        else:
            return sorted(similars_list, key=lambda x: x[1], reverse=True)[0]

    def Harmonic_Mean(self,ssim,distance):
        HM = 2 * (ssim * (1-distance)) / \
                    (ssim + (1-distance))

        return HM
