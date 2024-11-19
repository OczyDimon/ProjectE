import numpy as np
import matplotlib.pyplot as plt


numbers64 = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'


def convert_10_into_64(x: int) -> str:
    y = ''
    while x:
        y += numbers64[x % 64]
        x //= 64
    return y


def convert_64_into_10(x: str) -> int:
    y = 0
    for i in range(len(x)):
        y += numbers64.index(x[i]) * (64**i)
    return y


def convert_vectors_10_into_64cod(vv: list) -> str:  # шифрование вектора в 64ричную строку
    sl = ''
    for v in vv:
        if v > 0:
            vi = str(int(v//(0.1**16)))
            vi = '2' + '0' * (17 - len(vi)) + vi  # сохраняем знак + ввиде цифры 2 перед числом
        else:
            vi = str(int(abs(v)//(0.1**16)))
            vi = '1' + '0' * (17 - len(vi)) + vi  # сохраняем знак - ввиде цифры 1 перед числом
        sl += vi  # получаем строку-число унифицированной формы вида скейка(чётность{2/1} + число без точки)
    sl = convert_10_into_64(int(sl))
    return sl


def convert_64cod_into_vectors_10(s: str) -> list:  # дешифрование 64ричной строки в вектор шума
    sgn = {'2': 1, '1': -1}
    k = str(convert_64_into_10(s))
    v = []
    for j in range(100):
        prv = k[18*j: 18*(1+j)]  # берём по 18 символов отведённые на каждое значение
        prv = sgn[prv[0]] * (np.float64(prv[1:]) * (0.1**16))  # вытаскиваем знак
        # print(prv)
        v.append(prv)
    return v


def generate_by_vector(v, gen):
    #  конвертим под подходящий тип вектор
    converted_v0 = np.array(v)
    converted_v = np.array(np.ndarray(shape=(100,), buffer=converted_v0), ndmin=2)
    #  создаём кратинку
    generated_image = gen.predict(converted_v)[0]
    plt.figure(figsize=(10, 10))
    plt.imshow((generated_image * 0.5) + 0.5)  # Восстановление из нормализации
    plt.axis('off')  # почему через matlib? - самый удобный способ  преобразования
    plt.tight_layout()  # картинка возвращается в виде массива и проще всего её сделать там
    img_path = f'static/buffer/image_{str(v[1])[3:]}.png'
    plt.savefig(img_path)  # сохраняем загенеренную картинку
    return img_path


if __name__ == '__main__':
    z_dim = 100
    batch_size = 1
    noise = np.random.normal(0, 1, size=(batch_size, z_dim))
    print(noise[0])
    print(convert_64cod_into_vectors_10(convert_vectors_10_into_64cod(noise[0])))
    print(convert_64_into_10(convert_10_into_64(124)))
