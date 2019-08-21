from flask import Flask, request
from flask_cors import CORS
import _init_path
from models.conv import GatedConv
import sys
import numpy as np
import json
#import beamdecode

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route("/recognize", methods=["POST"])
def recognize():
    datas = request.json
    #datas = json.loads(request.get_data().decode("utf-8"))
    token = datas['token']
    receipt_data = list(datas['data'])
    if token == 'SR':
        model = GatedConv.load("SpeechRecognition_MASR/pretrained/gated-conv.pth")
        text = model.predict(receipt_data)
        print(text)
        return text
    elif token == 'FN':
        nums = np.array(receipt_data)
        mean = np.mean(nums)
        median = np.median(nums)
        return '平均数：{}   中位数：{}'.format(mean,median)


app.run("", debug=False)
