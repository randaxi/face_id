import telebot
import config
from model_train import get_predict
import os


def telegram_bot(token):
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id,
                         "Привет, это ГеленджикId.\n"
                         "Отправь мне фото для входа.\n")

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
                bot.send_sticker(message.chat.id, config.sticker_go_away)
            elif p == 1:
                bot.send_message(message.chat.id, "Заходи")
                bot.send_sticker(message.chat.id, config.sticker_come_in)
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
