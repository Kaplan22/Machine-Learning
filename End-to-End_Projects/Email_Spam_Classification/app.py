from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('email_classification.joblib')

# Define a route for predicting spam
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    prediction = model.predict([text])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
