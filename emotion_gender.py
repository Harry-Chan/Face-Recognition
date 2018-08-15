from keras.models import load_model
import cv2
import numpy as np


class emotion_gender(object):
    def __init__(self):
        self.gender_classifier = load_model(
            'gender_models/simple_CNN.81-0.96.hdf5', compile=False)

        self.gender_labels = {0: 'woman', 1: 'man'}

        self.emotion_classifier = load_model(
            'emotion_models/fer2013_mini_XCEPTION.107-0.66.hdf5', compile=False)

        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                               4: 'sad', 5: 'surprise', 6: 'neutral'}

    def gender_prediction(self, image):
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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
