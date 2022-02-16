import pandas as pd
import pickle
from flask import Flask,request

# Initialize app
apps = Flask(__name__)

# Load the pretrained classifier / Pickle file
open_pickle = open('classifier.pkl','rb')
clfr = pickle.load(open_pickle)

# Home page message
@apps.route('/')
def welcome():
	return "Welcome to Bank note authentication app"

# predict instance given manually
@apps.route('/predict')
def predict_note_instance():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    prediction = clfr.predict([[variance,skewness,curtosis,entropy]])
    
    return f"variance:{variance},\n skewness:{skewness},\n curtosis:{curtosis},\n entropy:{entropy}, \n Predicted score:{prediction}"

# predict instances provided in file
@apps.route('/predict_file')
def predict_file():
    test_df = pd.read_csv(request.files.get("file"))
    file_prediction = clfr.predict(test_df)
    return f"Prediction values for given .CSV file : {str(list(file_prediction))}"
 
# run app automatically
if __name__ == '__main__':
	apps.run()