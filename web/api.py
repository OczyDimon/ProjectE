import flask
from alg import *
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'kkkk'


def generate_by_vector(v, gen_path=f'../neural network/wsaves/generator_epoch_{2500}.h5'):
        #  конвертим под подходящий тип вектор
    converted_v0 = np.array(v)
    converted_v = np.array(np.ndarray(shape=(100,), buffer=converted_v0), ndmin=2)
        #  создаём кратинку
    gen = load_model(gen_path, compile=False)  # Берём нужную модель
    generated_image = gen.predict(converted_v)[0]
    plt.figure(figsize=(10, 10))
    plt.imshow((generated_image * 0.5) + 0.5)  # Восстановление из нормализации
    plt.axis('off')  # почему через matlib?
    plt.tight_layout()  # картинка возвращается в виде массива и проще всего её сделать там
    plt.savefig('buffer/image.jpg')  # сохраняем загенеренную картинку
    return True


@app.route('/')
def home():
    return '<img src="{{ url_for("static", filename="buffer/funimage.jpg") }}" alt="LoL">'

@app.route('/<string:code64>')
def generate(code64):
    if code64 and (code64 != 'favicon.ico'):
        try:
            os.remove('buffer/image.jpg')  # очищаем буфер от предыдущей картинки
        except Exception:
            pass
        v = convert_64cod_into_vectors_10(code64)
        generate_by_vector(v)  # создаём новую

        return '<img src="buffer\image.png">'


if __name__ == '__main__':
    app.run(debug=True)