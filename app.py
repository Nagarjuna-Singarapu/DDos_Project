import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('ddos_dt2.pkl', 'rb') as f:
    model = pickle.load(f) 

# Define the API endpoint
@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    pktcount = float(request.form['pktcount'])
    byteperflow = float(request.form['byteperflow'])
    pktperflow = float(request.form['pktperflow'])
    pktrate = float(request.form['pktrate'])
    tot_kbps = float(request.form['tot_kbps'])
    rx_kbps = float(request.form['rx_kbps'])
    flows = int(request.form['flows'])
    bytecount = float(request.form['bytecount'])
    Protocol = request.form['Protocol']
    tot_dur = float(request.form['tot_dur'])
    
    # Create a pandas DataFrame with the input data
    input_df = pd.DataFrame({
        'pktcount': [pktcount],
        'byteperflow': [byteperflow],
        'pktperflow': [pktperflow],
        'pktrate': [pktrate],
        'tot_kbps': [tot_kbps],
        'rx_kbps': [rx_kbps],
        'flows': [flows],
        'bytecount': [bytecount],
        'Protocol': [Protocol],
        'tot_dur': [tot_dur]
    })

    # Preprocess the input data
    cat_cols = ['Protocol']
    num_cols = input_df.columns.difference(cat_cols)
    scaler = preprocessing.StandardScaler()
    input_df[num_cols] = scaler.fit_transform(input_df[num_cols])
    input_df = pd.get_dummies(input_df, columns=cat_cols)

    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    prediction = np.random.randint(0, 1, size=1)

    print("Prediction:", prediction)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
