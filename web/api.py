import flask
from flask import render_template
import os
from alg import *
from keras.models import load_model
from shutil import rmtree
#import pyperclip


app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'kkkk'

gen_path = '../neural network/wsaves/generator_epoch_2500.h5'

gen = load_model(gen_path, compile=False)  # Берём нужную модель


@app.route('/')
def home():
    return render_template('Nonepage.html', img_path='static/buffer/funimage.jpg')
    #return render_template('<img src="{{ url_for("static", filename="static/buffer/funimage.jpg") }}" alt="LoL">')


@app.route('/generate')
def generate_base():
    noise_ = np.random.normal(0, 1, size=(1, 100))
    code_64 = convert_vectors_10_into_64cod(noise_[0])
    code_64 = f'http://127.0.0.1:5000/generate/{code_64}'
    return render_template('generate_base.html', code_64=code_64)


@app.route('/generate/<string:code64>')
def generate(code64):
    try:
        for image in os.listdir('static/buffer'):
            os.remove(f'static/buffer/{image}')
    except Exception as e:
        print(e)
    v = convert_64cod_into_vectors_10(code64)
    img_path = generate_by_vector(v, gen)  # создаём новую

    noise_ = np.random.normal(0, 1, size=(1, 100))
    code_64 = convert_vectors_10_into_64cod(noise_[0])
    code_64 = f'http://127.0.0.1:5000/generate/{code_64}'

    return render_template('generate.html', img_path=img_path.replace('static/', ''), code_64=code_64)


if __name__ == '__main__':
    app.run(debug=True)
