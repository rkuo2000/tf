# for AirDigit Recognizer App
# $./ngork http 5000 
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from tensorflow.keras import models

app = Flask(__name__)

labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'none']
model = models.load_model('models/airdigit_cnn.h5')
	
@app.route("/", methods=["GET"])
def hell():
    return "Hello!"
	
@app.route("/predict", methods=["POST"])	
def predict():
    keys = request.form.to_dict().keys()
    data = list(keys)[0][:-1].split(',')
    if len(data)<72: # patch data to 72
        for i in range(int((72-len(data))/3)):
            data = data + ['0.0','9.8','0.0'] # patch ax,ay,az = 0.0,9.8,0.0
    if len(data)>72: # cut data to 72 
       data = data[:72]
    x_test = np.asarray(data, dtype=float) # convert string list to float array
    x_test = x_test.reshape(-1,24,3) # reshape x_test
    preds = model.predict(x_test) # run model to predict
    pred = np.argmax(preds) # find maximum probility
    print(pred, labels[pred])
    return labels[pred]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
