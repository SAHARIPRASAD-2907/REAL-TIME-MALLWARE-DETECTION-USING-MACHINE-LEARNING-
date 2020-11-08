
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgboost_mallware_final', 'rb'))
@app.route('/',methods=['POST','GET'])
def home():
    return render_template('mallware_detection.html')

@app.route('/mallware_detection' ,methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        f = request.form["uplode-file"]
        data = pd.read_excel(f)
        X = data.drop(['Name', 'md5', 'legitimate'], axis = 1).values
        y = data['legitimate'].values
        y_pred = model.predict(X)
        if(y_pred[0]==0):
            	return render_template('mallware_detection.html', prediction_text='The file is not a mallware file')
        else:
            	return render_template('mallware_detection.html', prediction_text='The file has a mallware please dont install this file')



if __name__ == "__main__":
    app.run(debug=True)
