import telebot
import requests
import os
from alg_bot import *


TOKEN = "7953465629:AAH7OOq7U9-f-KYu9A1_BDsONhi1UNoJeGY"
bot = telebot.TeleBot(TOKEN)

API_URL = 'http://194.87.151.52:5000/api'


queue_bot = []

try:
    for image in os.listdir('buffer'):
        os.remove(f'buffer/{image}')
except Exception as e:
    print(e)


@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton(text='Получить аниме-девочку', callback_data='generate')
    markup.add(button)
    button = telebot.types.InlineKeyboardButton(text='Информация', callback_data='info')
    markup.add(button)
    bot.reply_to(message, "Привет! При помощи этого бота можно сгенерировать аниме-девочку."
                          " Жми на кнопку и смотри на результат", reply_markup=markup)


@bot.message_handler(commands=['info'])
def send_info(message):
    markup = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton(text='Вернуться в начало', callback_data='start')
    markup.add(button)
    bot.reply_to(
        message,
        "ProjectE - проект по дисциплине \"Технологии Программирования\","
        " суть которого генерация аниме девочек по запросу через веб клиент или телеграм бота. "
        "Проект реализовали:\n\n"
        "Дмитрий Митяков (Идейный вдохновитель проекта, автор большей части кода нейросети),\n"
        "Кирилл Сорокин (Разработка подключения веб интерфейса к нейросети),\n"
        "Кирилл Плетяго (Проектирование интерфейса для взаимодействия с нейросетью. Создание сайта и телеграм бота).",
        reply_markup=markup
    )


@bot.message_handler(commands=['generate'])
def generate_image(message):
    markup = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton(text='Получить аниме-девочку', callback_data='generate')
    markup.add(button)
    try:
        response = requests.get(API_URL).json()
        url = response['image']
        vector = response['vector']
        response_img = requests.get(url)

        img_path = f'buffer/image_{str(vector[0][1])[3:]}.png'
        
        if response_img.status_code == 200:
            with open(img_path, "wb") as file:
                file.write(response_img.content)

            with open(img_path, "rb") as photo:
                bot.send_photo(message.chat.id, photo, reply_markup=markup)

            update_buffer(queue_bot, img_path)

        else:
            bot.reply_to(message, f"Ошибка генерации изображения: {response_img.status_code}")
    except Exception as ex:
        bot.send_message(message.chat.id, f"Произошла ошибка: {ex}", reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.message:
        if call.data == 'generate':
            generate_image(call.message)
        if call.data == 'start':
            send_welcome(call.message)
        if call.data == 'info':
            send_info(call.message)


if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()
