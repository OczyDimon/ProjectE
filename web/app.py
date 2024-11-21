import flask
from flask import render_template
import os
from alg import *
from keras.models import load_model


app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'kkkk'

gen_path = '../neural network/wsaves/generator_epoch_2500.h5'

gen = load_model(gen_path, compile=False)  # Берём нужную модель

ip = ''

queue = []

try:
    for image in os.listdir('static/buffer'):
        os.remove(f'static/buffer/{image}')
except Exception as e:
    print(e)

@app.route('/')
def home():
    return '-_-'


@app.route('/generate')
def generate_base():
    noise_ = np.random.normal(0, 1, size=(1, 100))
    code_64 = convert_vectors_10_into_64cod(noise_[0])
    code_64 = f'http://127.0.0.1:5000/generate/{code_64}'
    return render_template('generate_base.html', code_64=code_64)


@app.route('/generate/<string:code64>')
def generate(code64):
    v = convert_64cod_into_vectors_10(code64)
    img_path = generate_by_vector(v, gen)  # создаём новую
    try:
        queue.append(img_path)
        if len(queue) > 4:
            g = queue.pop(0)
            os.remove(g)
    except Exception as e:
        print(e)

    noise = np.random.normal(0, 1, size=(1, 100))
    code_64 = convert_vectors_10_into_64cod(noise[0])
    code_64 = f'http://127.0.0.1:5000/generate/{code_64}'
    code64 = f'http://127.0.0.1:5000/generate/{code64}'

    return render_template('generate.html', img_path=img_path.replace('static/', ''), code_64=code_64, code64=code64)


@app.route('/api')
def api():
    noise = np.random.normal(0, 1, size=(1, 100))
    code_64 = convert_vectors_10_into_64cod(noise[0])
    v = convert_64cod_into_vectors_10(code_64)

    img_path = generate_by_vector(v, gen)

    try:
        queue.append(img_path)
        if len(queue) > 4:
            g = queue.pop(0)
            os.remove(g)
    except Exception as e:
        print(e)

    img_path = ip + img_path

    url = f'http://127.0.0.1:5000/generate/{code_64}'

    vector = noise.tolist()

    json_file = {
        'vector': vector,
        'code_64': code_64,
        'url': url,
        'image': img_path
    }

    return json_file


if __name__ == '__main__':
    app.run(debug=True)
