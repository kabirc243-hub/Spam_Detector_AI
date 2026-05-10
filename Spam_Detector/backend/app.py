from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle


app = Flask(__name__)


CORS(app)


model = pickle.load(open("model.pkl", "rb"))


vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return "Spam Detection API is running."


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data.get("message", "")

    
    message_vector = vectorizer.transform([message])

    
    probability = model.predict_proba(message_vector)[0]


    spam_probability = probability[1] * 100


    prediction = model.predict(message_vector)[0]


    result = "Spam" if prediction == 1 else "Not Spam"


    return jsonify({
        "prediction": result,
        "confidence": round(spam_probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)