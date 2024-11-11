import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import cv2
x = input()

generator = load_model(f'saves/generator_epoch_{x}0.h5')

z_dim = 100


def generate_and_display_images(generator, num_images=10):
    noise = np.random.normal(0, 1, size=(num_images, z_dim))
    generated_images = generator.predict(noise)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow((generated_images[i] * 0.5) + 0.5)  # Восстановление из нормализации
        plt.axis('off')
    plt.tight_layout()
    plt.show()


generate_and_display_images(generator)
