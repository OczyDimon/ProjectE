import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
from keras.optimizers import Adam

from datasets import load_dataset

# Загрузка датасета (5к изображений аниме девочек)
ds = load_dataset("jlbaker361/anime_faces_5k", split="train")


# Преобразование изображений
def preprocess_image(image):
    image = tf.image.resize(image, [64, 64])  # Изменение размера (изначально 256х256 -> 64х64)
    image = (image - 127.5) / 127.5  # Нормализация (все значения матрицы принимают значения от -1 до 1)
    return image


# Создание numpy массива из загруженных данных
images = []

for i, item in enumerate(ds):
    img = item['image']
    images.append(preprocess_image(img).numpy())
    if i % 100 == 0:
        print(f"Обработано {i} изображений")


image_data = np.array(images)


# Определение архитектуры GAN
# Генератор - полносвязная сеть, принимает вектор шума и генерирует изображение
def build_generator(latent_dim):
    model = Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(64 * 64 * 3, activation='tanh'),
        layers.Reshape((64, 64, 3))
    ])
    return model


# Дискриминатор - полносвязная сеть, принимает изображение
# и возвращает значение (от 0 до 1) "реалистичности" изображения

def build_discriminator():
    model = Sequential([
        layers.Flatten(input_shape=(64, 64, 3)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


# Создание генератора и дискриминатора, размер латентного пространства = 100
z_dim = 100
generator = build_generator(z_dim)
discriminator = build_discriminator()


discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

# Отключаем обучение дискриминатора и собираем модель GAN, добавляем входной слой - в него поступает вектор шума
gan_input = layers.Input(shape=(z_dim, ))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# Тренировка GAN
def train_gan(epochs=10000, batch_size=32, save_interval=100):
    for epoch in range(epochs+1):
        # Генерация случайного шума - нормальное распределение со средним = 0 и дисперсией = 1
        # размера количества поступающих изображений и вектора шума
        noise = np.random.normal(0, 1, size=(batch_size, z_dim))
        # Здесь создаются изображения
        generated_images = generator.predict(noise)

        # Выборка изображений из датасета для сравнения в количестве batch_size
        idx = np.random.randint(0, image_data.shape[0], batch_size)
        real_images = image_data[idx]

        # Обучение дискриминатора, подаются настоящие изображения, которые должны придти к значениям 1
        # Обучение дискриминатора, подаются сгенерированные изображения, которые должны придти к значениям 0
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))

        # Вычисление среднего качества оценки дискриминатора
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Обучение генератора, его цель привести к значению 1
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Вывод информации раз в 10 эпох и сохранение раз в save_interval эпох
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        if epoch % save_interval == 0:
            generator.save(f"wsaves/generator_epoch_{epoch}.h5")
            discriminator.save(f"wsaves/discriminator_epoch_{epoch}.h5")
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")


# Вызываем функцию обучения, задаем количество эпох, количество изображений в каждой эпохе и интервал сохранения
train_gan(epochs=3000, batch_size=32, save_interval=500)
