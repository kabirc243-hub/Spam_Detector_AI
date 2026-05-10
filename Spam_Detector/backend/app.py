from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return "Spam Detection API is running."


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data.get("message", "")

    # Convert message into vector
    message_vector = vectorizer.transform([message])

    # Predict probabilities
    probability = model.predict_proba(message_vector)[0]

    # Spam confidence percentage
    spam_probability = probability[1] * 100

    # Final prediction
    prediction = model.predict(message_vector)[0]

    # Convert numeric output to text
    result = "Spam" if prediction == 1 else "Not Spam"

    # Return JSON response
    return jsonify({
        "prediction": result,
        "confidence": round(spam_probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)