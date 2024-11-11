import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
from keras.optimizers import Adam

from datasets import load_dataset

# Загрузка датасета
ds = load_dataset("jlbaker361/anime_faces_5k", split="train")


# Преобразование изображений
def preprocess_image(image):
    image = tf.image.resize(image, [64, 64])  # Изменение размера
    image = (image - 127.5) / 127.5  # Нормализация
    return image


# Создание numpy массива из загруженных данных
images = []

for i, item in enumerate(ds):
    img = item['image']
    images.append(preprocess_image(img).numpy())
    if i % 100 == 0:
        print(f"Обработано {i} изображений")
    if i % 100 == 0:
        break

image_data = np.array(images)


# Определение архитектуры GAN
def build_generator(z_dim):
    model = Sequential([
        layers.Dense(256, input_dim=z_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(64 * 64 * 3, activation='tanh'),
        layers.Reshape((64, 64, 3))
    ])
    model = Sequential([
        layers.Dense(8*8*1024, input_shape=(z_dim,)),
        layers.Reshape((8, 8, 1024)),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(filters=512, kernel_size=4, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same'),
        layers.Activation('tanh')
    ])
    return model


def build_discriminator():
    model = Sequential([
        layers.Flatten(input_shape=(64, 64, 3)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    # model = Sequential([
    #     layers.Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 3), padding="same"),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dropout(0.25),
    #     layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
    #     layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
    #     layers.BatchNormalization(momentum=0.8),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dropout(0.25),
    #     layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
    #     layers.BatchNormalization(momentum=0.8),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dropout(0.25),
    #     layers.Conv2D(256, kernel_size=3, strides=1, padding="same"),
    #     layers.BatchNormalization(momentum=0.8),
    #     layers.LeakyReLU(alpha=0.2),
    #     layers.Dropout(0.25),
    #     layers.Flatten(),
    #     layers.Dense(1, activation='sigmoid')
    # ])
    model = Sequential([
        layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3)),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(filters=1024, kernel_size=4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Flatten(),
        layers.Dense(1),
        layers.Activation('sigmoid')
    ])
    return model

# Компиляция моделей
z_dim = 100
generator = build_generator(z_dim)
discriminator = build_discriminator()


discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

# Объединенная модель GAN
gan_input = layers.Input(shape=(z_dim, ))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# Тренировка GAN
def train_gan(epochs=10000, batch_size=32, save_interval=100, count_interval=10):
    for epoch in range(epochs):
        # Генерация случайного шума
        noise = np.random.normal(0, 1, size=(batch_size, z_dim))
        generated_images = generator.predict(noise)

        # Получение реальных изображений
        idx = np.random.randint(0, image_data.shape[0], batch_size)
        real_images = image_data[idx]

        # Создание меток
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Обучение генератора
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Сохранение модели
        if epoch % count_interval == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        if epoch % save_interval == 0:
            generator.save(f"saves/generator_epoch_{epoch}.h5")
            discriminator.save(f"saves/discriminator_epoch_{epoch}.h5")
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")


train_gan(epochs=10000, batch_size=32, save_interval=500, count_interval=10)
