import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

# Размер латентного вектора
z_dim = 100


def generate_and_display_images(generator, num_images=10):
    # Генерируем изображения по принципу, использованному в обучениия
    noise = np.random.normal(0, 1, size=(num_images, z_dim))
    generated_images = generator.predict(noise)

    # Вывод на экран
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow((generated_images[i] * 0.5) + 0.5)  # Восстановление из нормализации
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Запуск разных моделей с целью проверки
for count in range(7):
    generator = load_model(f'wsaves/generator_epoch_{500*count}.h5', compile=False)
    generate_and_display_images(generator)
