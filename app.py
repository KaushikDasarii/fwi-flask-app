from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ridge.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features from form input
            features = [
                float(request.form['temp']),
                float(request.form['rh']),
                float(request.form['ws']),
                float(request.form['rain'])
            ]
            features = np.array(features).reshape(1, -1)

            # Scale and predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
