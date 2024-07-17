from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('data', 'fish_weight_model.pkl')
model = joblib.load(model_path)

port = int(os.environ.get('PORT', 5000))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(abs(prediction[0]), 2)  # Using abs() to ensure non-negative prediction

        input_values = dict(request.form)  # Capture input values

        return render_template(
            'index.html',
            prediction_text=f'Predicted Fish Weight: {output} grams',
            input_values=input_values  # Pass input values to the template
        )
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

