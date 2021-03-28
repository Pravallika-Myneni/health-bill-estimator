# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# API definition
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'success', 'data': None, 'message': 'Welcome to the prediction API'})

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            
            age = json_.get("age")
            sex = json_.get("sex")
            bmi = json_.get("bmi")
            children = json_.get("children")
            smoker = json_.get("smoker")
            region = json_.get("region")

            user_input = {'age' : [age], 'sex' : [sex] , 'bmi' : [bmi] , 'children' : [children] ,'smoker': [smoker], 'region': [region] }
            user_input_df = pd.DataFrame(user_input)

            user_input_df['sex'] = user_input_df['sex'].apply(lambda x: 1 if x=='female' else 0)
            user_input_df['smoker'] = user_input_df['smoker'].apply(lambda x: 1 if x=='yes' else 0)
            user_input_df['region'] = user_input_df['region'].apply(lambda x: 3 if x=='northeast' else ( 2 if x== 'northwest' else (1 if x== 'southeast' else 0)))
            user_ip = user_input_df.drop(columns= ['sex','region','children'])

            predicted_cost = lr.predict(user_ip)

            return jsonify({'status': 'success', 'data': str(predicted_cost[0]), 'message': 'Cost successfully predicted'})

        except:

            return jsonify({'status': 'error', 'data': None, 'message': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    # port definition for development environment
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model.pkl") # Load "model.pkl"
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"

    app.run(port=port, debug=True)