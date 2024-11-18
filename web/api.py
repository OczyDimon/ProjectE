import flask
from alg import convert_64cod_into_vectors_10
import os
#from work import generate_by_vector

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'kkkk'

@app.route('/')
def home():
    return '<img src="{{ url_for("static", filename="buffer/funimage.jpg") }}" alt="LoL">'

@app.route('/<string:code64>')
def generate(code64):
    """
    try:
        os.remove('buffer/image.jpg')
    except Exception:
        pass
    v = convert_64cod_into_vectors_10(code64)
    generate_by_vector(v, 'buffer/image.jpg')
    """
    return '<img src="buffer\image.jpg">'


if __name__ == '__main__':
    app.run(debug=True)