# $./ngork http 5000 
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

from tensorflow.keras import models

app = Flask(__name__)

model = models.load_model('models/airdigit_cnn.h5')
	
@app.route("/", methods=["GET"])
def hell():
    return "Hello!"
	
@app.route("/predict", methods=["POST"])	
def predict():
    f = request.files['file']
    df = pd.read_csv(f, delimiter=',', header=None)
    x_test = np.array(df.iloc[0])
    x_test = x_test[:-1]# remove last one (nan)
    x_test = x_test.reshape(-1,24,3)

    preds = model.predict(x_test)
    pred = np.argmax(preds)
    print(pred)
    return str(pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
