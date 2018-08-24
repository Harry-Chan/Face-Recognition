from keras.models import load_model
import cv2
import numpy as np


class emotion_gender(object):
    def __init__(self):
        # loading model

        # 性別模組
        # 可用模組 gender_mini_XCEPTION.21-0.95.hdf5、simple_CNN.81-0.96.hdf5
        self.gender_classifier = load_model(
            'gender_models/simple_CNN.81-0.96.hdf5', compile=False)

        self.gender_labels = {0: 'woman', 1: 'man'}

        # 表情模組
        # 可用模組 emotion_model.hdf5 、 fer2013_mini_XCEPTION.107-0.66.hdf5 、simple_CNN.985-0.66.hdf5
        self.emotion_classifier = load_model(
            'emotion_models/simple_CNN.985-0.66.hdf5', compile=False)

        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                               4: 'sad', 5: 'surprise', 6: 'neutral'}

    def gender_prediction(self, image):
        # 使用 simple_CNN.985-0.66.hdf5，模組其input維度為4維，原先3維必須在擴增
        # 使用 gender_mini_XCEPTION.21-0.95.hdf5，必須先轉灰階在擴增維4維
        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 讀取模組input的size
        gender_input_size = self.gender_classifier.input_shape[1:3]
        # 將圖片resize
        image = cv2.resize(image, (gender_input_size))
        # 同除255正規化
        image = image / 255

        # 在0的位置擴增維度 (64,64,3) 變成 (1,64,64,3)
        image = np.expand_dims(image, 0)
        #image = np.expand_dims(image, -1)

        gender_prediction = self.gender_classifier.predict(image)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = self.gender_labels[gender_label_arg]
        print('gender', gender_text)

        return gender_text

    def emotion_prediction(self, image):
        # 表情模組必須先轉成灰階，input為4個維度

        # 轉灰階
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 讀取模組input的size
        emotion_input_size = self.emotion_classifier.input_shape[1:3]
        # 將圖片resize
        gray_image = cv2.resize(gray_image, (emotion_input_size))
        # 同除255正規化
        gray_image = gray_image/255

        # 在0的位置擴增維度
        gray_image = np.expand_dims(gray_image, 0)
        # 在-1的位置擴增維度
        gray_image = np.expand_dims(gray_image, -1)

        emotion_prediction = self.emotion_classifier.predict(gray_image)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.emotion_labels[emotion_label_arg]
        print('emotion', emotion_text)

        return emotion_text
