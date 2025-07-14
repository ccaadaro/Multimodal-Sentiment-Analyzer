from flask import Flask, render_template, request
import requests
import base64
from prometheus_flask_exporter import PrometheusMetrics
import time

app = Flask(__name__)
metrics = PrometheusMetrics(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    image = request.files['image']
    text = request.form['text']
    img_bytes = image.read()

    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': ('image.jpg', img_bytes, image.content_type)},
        data={'text': text}
    )

    prediction = response.json()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    app.logger.info(f"Inference time: {time.time() - start_time:.4f} seconds")

    return render_template('index.html', prediction=prediction, img_data=img_b64, user_text=text)