
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

@app.route("/")
def home():
    return "ML Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])

    return jsonify({
        "prediction": int(prediction[0]),
        "class_name": iris.target_names[prediction[0]]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)