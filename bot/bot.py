import telebot
import requests


TOKEN = "7953465629:AAH7OOq7U9-f-KYu9A1_BDsONhi1UNoJeGY"
bot = telebot.TeleBot(TOKEN)

API_URL = "http://194.87.151.52:5000/generate"  # Вставить адрес, где генерируется аниме-девочка, иначе -вайб


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! При помощи этого бота можно сгенерировать аниме-девочку."
                          " Пиши /generate и смотри на результат")


@bot.message_handler(commands=['info'])
def send_info(message):
    bot.reply_to(
        message,
        "ProjectE - проект по дисциплине \"Технологии Программирования\","
        " суть которого генерация аниме девочек по запросу через веб клиент или телеграм бота. "
        "Проект реализовали:\n\n"
        "Дмитрий Митяков (Идейный вдохновитель проекта, автор большей части кода нейросети),\n"
        "Кирилл Сорокин (Разработка подключения веб интерфейса к нейросети),\n"
        "Кирилл Плетяго (Проектирование интерфейса для взаимодействия с нейросетью. Создание сайта и телеграм бота)."
    )


@bot.message_handler(commands=['generate'])
def generate_image(message):
    try:
        response = requests.get(API_URL)
        
        if response.status_code == 200:
            with open("generated_image.jpg", "wb") as file:
                file.write(response.content)
            
            with open("generated_image.jpg", "rb") as photo:
                bot.send_photo(message.chat.id, photo)
        else:
            bot.reply_to(message, f"Ошибка генерации изображения: {response.status_code}")
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")


if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()
