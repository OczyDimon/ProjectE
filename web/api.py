import flask
from flask import render_template
from alg import *
from keras.models import load_model


app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'kkkk'

gen_path = '../neural network/wsaves/generator_epoch_2500.h5'

gen = load_model(gen_path, compile=False)  # Берём нужную модель


@app.route('/')
def home():
    return render_template('<img src="{{ url_for("static", filename="buffer/funimage.jpg") }}" alt="LoL">')


@app.route('/generate/<string:code64>')
def generate(code64):
    v = convert_64cod_into_vectors_10(code64)
    img_path = generate_by_vector(v, gen)  # создаём новую

    return render_template('generate.html', img_path=img_path.replace('static/', ''))


if __name__ == '__main__':
    app.run(debug=True)
