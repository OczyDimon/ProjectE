import numpy as np
import tensorflow as tf
from keras import layers, Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, BatchNormalization, Activation,\
    UpSampling2D, Reshape, Dropout, Flatten, ZeroPadding2D, LeakyReLU

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


image_data = np.array(images)


# Определение архитектуры GAN
def build_generator(latent_dim):
    model = Sequential()

    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=latent_dim))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()

    return model


def build_discriminator():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64, 64, 3),
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# Компиляция моделей
z_dim = 128
generator = build_generator(z_dim)
discriminator = build_discriminator()


discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False

# Объединенная модель GAN
gan_input = layers.Input(shape=(z_dim, ))
print(gan_input)
generated_image = generator(gan_input)
print(generated_image)
gan_output = discriminator(generated_image)
print(gan_output)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

d_loss_sum = []
g_loss_sum = []


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

        d_loss_sum.append(d_loss)
        g_loss_sum.append(g_loss)

        # Сохранение модели
        if epoch % count_interval == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        if epoch % save_interval == 0:
            generator.save(f"saves/generator_epoch_{epoch}.h5")
            discriminator.save(f"saves/discriminator_epoch_{epoch}.h5")
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")


train_gan(epochs=1000, batch_size=32, save_interval=10, count_interval=5)

np.save('d_loss.npy', np.array(d_loss_sum))
np.save('g_loss.npy', np.array(g_loss_sum))
