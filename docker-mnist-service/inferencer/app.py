
from flask import Flask, Blueprint, request, jsonify
from werkzeug.exceptions import abort
from inferencer import classify


import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)


@app.route('/classify', methods=['POST'], strict_slashes=False)
def classify_endpoint():
    imagefile = request.files.get('imagefile', '')
    prediction = classify(img=imagefile)
    return jsonify({
        "predicted_digit": str(prediction)
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0')
