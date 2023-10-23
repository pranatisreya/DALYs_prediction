from flask import Flask, render_template, request
import numpy as np
import joblib
import os 

app = Flask(__name__)

# Load the pre-trained model
path=os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(path,'DALYs.pkl'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def prediction():
    # Get input values from user
    ct= int(request.form['Country'])
    yr = int(request.form['Years'])
    sp = float(request.form['Schizophrenia'])
    bd = float(request.form['Bipolar_Disorder'])
    ed = float(request.form['Eating_Disorder'])
    anx = float(request.form['Anxiety'])
    dg = float(request.form['Drug_Addict'])
    dep = float(request.form['Depression'])
    alc= float(request.form['Alcoholism'])

    # Create input data array
    input_data = np.array([[ct,yr,sp,bd,ed,anx,dg,dep,alc]])

    # Predict the price using the pre-trained model
    predicted_dayls  = model.predict(input_data)[0]

    # Render the result page with predicted price
    return render_template('index.html', predicted_dayls=predicted_dayls)

if __name__ == '__main__':
    app.run(debug=True)
