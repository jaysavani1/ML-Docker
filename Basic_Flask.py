import pandas as pd
import pickle
from flask import Flask,request


apps = Flask(__name__)

open_pickle = open('classifier.pkl','rb')
clfr = pickle.load(open_pickle)

@apps.route('/')
def welcome():
	return "Welcome to Bank note authentication app"

@apps.route('/predict')
def predict_note_instance():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    prediction = clfr.predict([[variance,skewness,curtosis,entropy]])
    
    return f"variance:{variance},\n skewness:{skewness},\n curtosis:{curtosis},\n entropy:{entropy}, \n Predicted score:{prediction}"

if __name__ == '__main__':
	apps.run()