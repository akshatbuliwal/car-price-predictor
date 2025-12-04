from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Correct path for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "LinearRegressionModel.pkl"), "rb"))
onehot = pickle.load(open(os.path.join(BASE_DIR, "OneHotEncoder.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "StandardScaler.pkl"), "rb"))
data = pd.read_csv(os.path.join(BASE_DIR, "Cleaned_Car_data.csv"))

@app.route("/options", methods=["GET"])
def get_dropdown_options():
    companies = sorted(data["company"].unique())
    models_by_company = {
        company: sorted(data[data["company"] == company]["name"].unique())
        for company in companies
    }
    years = sorted([int(y) for y in data["year"].unique()])
    fuel_types = sorted(data["fuel_type"].unique())
    return jsonify({
        "companies": companies,
        "models_by_company": models_by_company,
        "years": years,
        "fuel_types": fuel_types
    })

@app.route("/predict", methods=["POST"])
def predict_price():
    content = request.json
    try:
        company = content["company"]
        name = content["name"]
        year = int(content["year"])
        fuel_type = content["fuel_type"]
        kms_driven = int(content["kms_driven"])

        query_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                columns=["name", "company", "year", "kms_driven", "fuel_type"])

        transformed_cat = onehot.transform(query_df[["name", "company", "fuel_type"]])
        scaled_num = scaler.transform(query_df[["year", "kms_driven"]])
        final_input = np.hstack((transformed_cat.toarray(), scaled_num))

        predicted_price = model.predict(final_input)[0]
        return jsonify({"estimated_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Car Price Predictor API Works!"

if __name__ == "__main__":
    app.run(host="0.0.0.0")
