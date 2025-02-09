from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_disease_prediction_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Prepare input data for prediction
        user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
        prediction = model.predict(user_data)

        # Determine the result
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease Present"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
