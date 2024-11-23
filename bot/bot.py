import telebot

TOKEN = "7953465629:AAH7OOq7U9-f-KYu9A1_BDsONhi1UNoJeGY"
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! При помощи этого бота можно сгенерировать аниме-девочку. Пиши /generate и смотри на результат")

@bot.message_handler(commands=['info'])
def send_info(message):
    bot.reply_to(
        message,
        "ProjectE - проект по дисциплине \"Технологии Программирования\", суть которого генерация аниме девочек по запросу через веб клиент или телеграм бота. "
        "Проект реализовали:\n\n"
        "Дмитрий Митяков (Идейный вдохновитель проекта, автор большей части кода нейросети),\n"
        "Кирилл Сорокин (Разработка подключения веб интерфейса к нейросети),\n"
        "Кирилл Плетяго (Проектирование интерфейса для взаимодействия с нейросетью. Создание сайта и телеграм бота)."
    )

# Добавить /generate


if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()
