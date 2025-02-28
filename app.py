from flask import Flask, render_template, request
import os
from masking_model import run_mask_prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    images = []
    if request.method == 'POST':
        input_text = request.form.get('input_text', '')
        preds, imgs = run_mask_prediction(input_text)
        if preds is not None:
            predictions = preds
            images = imgs
    return render_template('index.html', predictions=predictions, images=images)

if __name__ == '__main__':
    # Ensure the static/generated-images folder exists
    images_path = os.path.join('static', 'generated-images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    app.run(debug=True)
