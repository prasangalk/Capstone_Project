import numpy as np
import pandas as pd

# Classifier algorithms
from sklearn.ensemble import RandomForestRegressor

from flask import Flask, jsonify, request

import joblib
import json
    
def pre_processing(data_input):    
    
    data_input['is_smoker'] = np.where((data_input['smoker'] == 'yes') , 1,0)
    data_input['is_male'] = np.where((data_input['sex'] == 'male') , 1,0)

    #Adding One-hot encoding
    data_input[['reg_northeast','reg_northwest','reg_southeast','reg_southwest']] = 0
    #One hot encoding considering the region column
    if (data_input['region'].str.contains('northeast').any):
        data_input['reg_northeast'] = 1
    elif (data_input['region'].str.contains('northwest').any):
        data_input['reg_northwest'] = 1
    elif (data_input['region'].str.contains('southeast').any):
        data_input['reg_southeast'] = 1
    elif (data_input['region'].str.contains('southwest').any):
        data_input['reg_southwest'] = 1

    #data_input = data_input.drop(columns= ['reg_southeast','charges'])
    if (data_input['region'].str.contains('reg_southeast').any):
        data_input.drop(columns= ['reg_southeast'])

    dataset_output = data_input[['age','bmi','children','is_smoker','is_male','reg_northeast','reg_northwest','reg_southwest']]
    return dataset_output  

def score(input_data, model):
    return model.predict(input_data)

app = Flask(__name__)

# Load model
model_file = 'Model_RFR.joblib'
model = joblib.load(model_file)
print(model)
    
@app.route("/")
def index():
    return "Greetings from Prediction API"

@app.route("/regressor", methods=['POST'])
def regressor():
    if request.method == 'POST': 
        input_data =  request.form.to_dict()
        print(input_data)
        #print(type(input_data))
        input_data = pd.DataFrame([input_data])
        #print(input_data)
        pre_prossed_data=pre_processing(input_data)
        #print(pre_prossed_data)
        test_prediction = score(pre_prossed_data, model)
        print(test_prediction[0])
        prediction = test_prediction[0] 
        return jsonify({'prediction': prediction})  
        
if __name__ == '__main__':
    app.run(debug=True, port=5001)