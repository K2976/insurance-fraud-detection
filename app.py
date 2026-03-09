from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    age = int(request.form["age"])
    months = int(request.form["months"])
    claim = float(request.form["claim"])
    severity = int(request.form["severity"])

    features = np.array([[age,months,claim,severity]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Fraudulent Claim"
    else:
        result = "Genuine Claim"

    return render_template("index.html",prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)