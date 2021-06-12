from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score  # классы сбалансированы
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


X = (np.load('data.npy')).tolist()
print('X downloaded')

y = (np.load('data_y.npy')).tolist()
out_encoder = LabelEncoder()
out_encoder.fit(y)
y = out_encoder.transform(y)
print('y downloaded')

embedder = FaceNet()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=179
)

model = SVC(kernel='poly', random_state=179)
model.fit(X_train, y_train)
print('model fitted')

# # predict
# yhat_train = model.predict(X_train)
# yhat_test = model.predict(X_test)
#
# # score
# score_train = accuracy_score(y_train, yhat_train)
# score_test = accuracy_score(y_test, yhat_test)
#
# # precision
# prec_train = precision_score(y_train, yhat_train)
# prec_test = precision_score(y_test, yhat_test)
#
# # recall
# rec_train = recall_score(y_train, yhat_train)
# rec_test = recall_score(y_test, yhat_test)
#
# # summarize
# print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
# print('Precision: train=%.3f, test=%.3f' % (prec_train * 100, prec_test * 100))
# print('Recall: train=%.3f, test=%.3f' % (rec_train * 100, rec_test * 100))


def get_predict(path_to_image):
    try:
        global model
        detections = embedder.extract(path_to_image, threshold=0.95)
        emb = detections[0]['embedding']

        if len(detections) == 1:
            return model.predict(Normalizer(norm='l2').transform([emb]))
        else:
            return -1  # 'На фото больше одного лица. Кто в итоге хочет пройти?'
    except IndexError:
        return -2  # 'На изображении плохо видно лицо\nСделай, пожалуйста, новую фотографию :)'
