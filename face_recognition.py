import dlib
import numpy as np
import cv2
from face_aligner import FaceAligner
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr


class face_recognition(object):
    def __init__(self):
        # loading model

        # 人臉偵測模型(cnn)
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            './models/mmod_human_face_detector.dat')

        # 人臉偵測模型(hog)
        self.face_detector = dlib.get_frontal_face_detector()

        # 人臉特徵模型(5點)
        self.pose_predictor_5_point = dlib.shape_predictor(
            './models/shape_predictor_5_face_landmarks.dat')

        # 人臉特徵模型(68點)
        # self.pose_predictor_68_point = dlib.shape_predictor(
        #    './models/shape_predictor_68_face_landmarks.dat')

        # 人臉向量模型
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

    def face_detection(self, img, number_of_times_to_upsample=1, model="cnn"):
        # 使用dlib的face_detecctor偵測人臉位置，number_of_times_to_upsample代表採樣次數
        face_locations = []
        if model != "cnn":
            faces = self.face_detector(img, number_of_times_to_upsample)
            for face in faces:
                if self.bounds(face, img.shape) == True:
                    face_locations.append(
                        (face.top(), face.right(), face.bottom(), face.left()))
            return face_locations
        else:
            faces = self.cnn_face_detector(img, number_of_times_to_upsample)
            for face in faces:
                # 判斷是否超出畫面與偵測出的信心度有無大於1
                if self.bounds(face.rect, img.shape) == True and face.confidence > 1:
                    # 將位置存成(上,又,下,左)的tuple
                    face_locations.append(
                        (face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()))
            return face_locations

    def face_encodings(self, image, face_locations=None, num_jitters=1):
        # 將人臉編碼成128維的向量，num_jitters代表採樣次數
        # Given an image, return the 128-dimension face encoding for each face in the image.
        pose_predictor = self.pose_predictor_5_point

        # 當 face_locations is None 為讀取資料夾的圖片
        if face_locations is None:
            cv2.imshow('known_face', image)

            # 邊界即為圖片大小
            bottom, right, _ = image.shape
            face_location = (0, right, bottom, 0)

            # 獲取特徵點
            raw_landmark = pose_predictor(
                image, self._css_to_rect(face_location))

            # 印出特徵點
            # print("---"*5, raw_landmarks[0].part(0),
            #       raw_landmarks[0].part(1), raw_landmarks[0].part(2), raw_landmarks[0].part(3), raw_landmarks[0].part(4))

            # 帶入人臉向量模型並回傳其128維向量空間
            return np.array(self.face_encoder.compute_face_descriptor(image, raw_landmark, num_jitters))

        # 計算偵測到的人臉向量
        else:
            raw_landmarks = []
            faces_images = []
            face_encodings = []
            for face_location in face_locations:
                top, right, bottom, left = face_location
                faces_image = image[top:bottom, left:right]

                # 將人臉圖像都resize成 200 * 200
                faces_image = cv2.resize(faces_image, (200, 200))
                faces_images.append(faces_image)

                # 將圖像計算特徵點
                raw_landmark = pose_predictor(
                    faces_image, self._css_to_rect((0, 200, 200, 0)))
                raw_landmarks.append(raw_landmark)

                # 印出特徵點
                test = cv2.resize(image[top:bottom, left:right], (200, 200))
                for i in range(5):
                    cv2.circle(test, (raw_landmark.part(i).x,
                                      raw_landmark.part(i).y), 5, (0, 0, 255), -1)
                cv2.imshow('test', test)

                # 特徵點4為鼻子下方，藉此來判斷是否在畫面中間，排除抓到側臉與上仰的臉
                if raw_landmark.part(4).x > 145 or raw_landmark.part(4).x < 55 or raw_landmark.part(4).y < 105:

                    print("nose", raw_landmark.part(
                        4).x, raw_landmark.part(4).y)
                    # 側臉給空list
                    face_encodings.append([])
                    continue
                else:
                    face_encoding = np.array(self.face_encoder.compute_face_descriptor(
                        faces_image, raw_landmark, num_jitters))
                    face_encodings.append(face_encoding)

            return face_encodings, faces_images

    # def face_aligners(self, face_image, face_locations):
    #     face_aligners = []

    #     for face_location in face_locations:

    #         face_aligner = self.face_aligner.align(
    #             face_image, face_image, self._css_to_rect(face_location))
    #         cv2.imshow('found_face', face_aligner)
    #         face_aligners.append(face_aligner)

    #     return face_aligners

    def _css_to_rect(self, css):
        # 將像素位置(top, right, bottom, left)轉換成dlib使用的格式
        # Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def compare_faces(self, face_encoding_to_check, face_location_to_check, people_object_list, ssim_threshold=0.7, dis_threshold=0.6):
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

            if similars_ssim >= ssim_threshold and encoding_distance <= dis_threshold:

                # 將兩個參數計算Harmonic Mean
                HM = self.Harmonic_Mean(similars_ssim, encoding_distance)
                similars_list.append((num, HM))
                print("OO"*5, (num, similars_ssim, encoding_distance))

            # 當兩張人臉在畫面上的距離<=30就放寬標準，其中一項符合就計算HM
            elif center_distance <= 30 and (similars_ssim >= ssim_threshold or encoding_distance <= dis_threshold):

                # 將兩個參數計算Harmonic Mean
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
            # 回傳HM最高的名稱
            return sorted(similars_list, key=lambda x: x[1], reverse=True)[0]

    def Harmonic_Mean(self, ssim, distance):
        # Harmonic_Mean兩個參數要越高算出來越大 (因為距離越短代表越相近所以用1減距離)
        HM = 2 * (ssim * (1-distance)) / (ssim + (1-distance))
        return HM
