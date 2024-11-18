import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

# Размер латентного вектора
z_dim = 100
noise1 = np.random.normal(0, 1, size=(1, z_dim))
print(type(noise1[0][1]))
noise2 = np.ndarray(np.ndarray(list(map(np.float64,'1.4328436033320924 -2.4673949113193956 -0.07095104360925833 1.0039061753779814 1.170123890043556 1.6583707974176984 -0.24457554826452066 -0.35817491468664325 -1.9896066405985144 -0.1146832037000868 -0.7675106625536197 -1.219922102010227 -1.389068302786585 -0.24458346031126485 -0.41478865891666217 -0.4820078208643363 -0.7989691447803003 -0.05829408388171912 -1.4034322858679606 1.7556845974616535 0.5578854702204455 -1.9521522056004017 0.5943555254904906 0.42406389762615737 -1.9658862347400603 0.5734551776012052 -1.3512763078100387 0.24974767195671968 -1.2449410161152925 -1.0983383170923484 -0.4119180728607392 -0.6726977594896364 2.2276508072148546 -0.07968203281101546 0.8201990189440673 1.4272925405217995 0.1295163952740725 -0.4475978935226734 0.691901481266421 -1.7608866065963351 0.40325939200664107 -0.015134393842411817 -1.133037805233324 0.3197913800028669 -0.6334431431061759 -1.640703530040248 0.36622337254332354 -0.9515842476325314 -1.698435375286686 1.2001973442235456 0.2560194664366474 2.788743937676616 -0.7838719336331532 0.07829823109011436 -0.6565952313220482 0.7482958366479151 -0.005205774926975323 -0.3226057149023912 -0.40341704861213973 -1.5455566015878568 -0.8966883956623953 0.2961369128479583 -0.5217951027494152 0.440669868056674 0.40281833695735203 1.4400798546164026 0.4022080261197449 -1.5655840552032079 -0.4828450982331217 -0.7816884156803999 -0.9836315193535043 0.9146491633953657 2.1194255980448222 0.505617912316704 2.8677155015386773 1.1165995933696367 0.27636743065175234 -0.06675874851825213 0.5089018331939014 -2.1321824054845013 -0.8410253310065049 -1.7014030298082752 0.4610111013706233 0.2434015847609197 0.7662823395497291 -1.9057840056966175 0.833024825161851 0.3864130641568973 -1.2149140917569392 0.6899748852638056 0.33382119831594625 -0.7093475710071219 0.38806219460083113 0.41670699668633426 -1.400412788003334 0.4405988921455801 0.4798578477139231 -0.7092274408290072 -1.3408818652760626 1.0040454992447283'.split()))))
print(noise2)

def generate_and_display_images(noise, generator, num_images=1):
    # Генерируем изображения по принципу, использованному в обучениия

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
#generator = load_model(f'../neural network/wsaves/generator_epoch_{2500}.h5', compile=False)
#generate_and_display_images(noise1, generator)
