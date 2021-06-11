from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score  # классы сбалансированы
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import telebot
import config
# import requests
# from pprint import pprint
import os

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

print(len(X_train), len(X_test), len(y_train), len(y_test))

model = SVC(kernel='poly', random_state=179)
model.fit(X_train, y_train)
print('model fitted')

# predict
yhat_train = model.predict(X_train)
yhat_test = model.predict(X_test)

# score
score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_test, yhat_test)

# precision
prec_train = precision_score(y_train, yhat_train)
prec_test = precision_score(y_test, yhat_test)

# recall
rec_train = recall_score(y_train, yhat_train)
rec_test = recall_score(y_test, yhat_test)

# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))
print('Precision: train=%.3f, test=%.3f' % (prec_train * 100, prec_test * 100))
print('Recall: train=%.3f, test=%.3f' % (rec_train * 100, rec_test * 100))


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


def telegram_bot(token):
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id,
                         "Привет, это ГеленджикId.\n"
                         "Отправь мне фото для входа.\n")

    # @bot.message_handler(content_types=["document"])
    # def temp_name(message):
    #     # pprint(message.json)
    #     # print(message.document)
    #     # print(message.document.file_id)
    #
    #     file_id = message.document.file_id
    #     file_info = bot.get_file(file_id)
    #
    #     bot.send_message(message.chat.id, "Спасибо, получил твоё фото. Сейчас, сейчас...")
    #
    #     r = requests.get('https://api.telegram.org/file/bot{0}/{1}'.format(token, file_info.file_path))
    #     open(file_info.file_path, 'wb').write(r.content)
    #
    #     # p = get_predict(file_info.file_path)
    #     # if p == 0:
    #     #     bot.send_message(message.chat.id, "Уходи")
    #     # elif p == 1:
    #     #     bot.send_message(message.chat.id, "Здравствуй, хозяин")
    #     # elif p == -1:
    #     #     bot.send_message(message.chat.id, "На фото больше одного лица. Кто в итоге хочет пройти?")
    #     # elif p == -2:
    #     #     bot.send_message(message.chat.id, "На изображении плохо видно лицо\nСделай, пожалуйста, новую фотографию :)")
    #     # os.remove(file_info.file_path)

    @bot.message_handler(content_types=["photo"])
    def get_answer(message):
        try:
            file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            bot.send_message(message.chat.id, "Спасибо, получил твоё фото. Сейчас, сейчас...")

            with open(file_info.file_path, 'wb') as new_file:
                new_file.write(downloaded_file)

            p = get_predict(file_info.file_path)
            if p == 0:
                # bot.send_message(message.chat.id, "Уходи")
                bot.send_sticker(message.chat.id,
                                 "CAACAgIAAxkBAAMeYMMZCQdCovxyzXft8oA2la_7j5oAAq0AA5FoDAAB5B_DTRv4450fBA")
            elif p == 1:
                bot.send_message(message.chat.id, "Заходи")
                bot.send_sticker(message.chat.id,
                                 "CAACAgIAAxkBAAMpYMMaENTZcewMuG3ZQCy4ZYPfH8oAAusDAAI6uRUCX6P4dFk_o2EfBA")
            elif p == -1:
                bot.send_message(message.chat.id, "На фото больше одного лица. Кто в итоге хочет пройти?")
            elif p == -2:
                bot.send_message(message.chat.id,
                                 "На изображении плохо видно лицо\nСделай, пожалуйста, новую фотографию :)")
            os.remove(file_info.file_path)
        except Exception as e:
            bot.reply_to(message, e)

    @bot.message_handler(func=lambda message: True,
                         content_types=['audio', 'video', 'text', 'location', 'contact', 'sticker'])
    def default_command(message):
        # pprint(message.json)
        bot.reply_to(message, "Давай без вот этого вот\nЖду твоих фотографий")

    bot.polling()


if __name__ == '__main__':
    telegram_bot(config.token)
