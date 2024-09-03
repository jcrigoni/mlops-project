from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
##
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
##
from dotenv import load_dotenv
load_dotenv()
import datetime

ARIZE_SPACE_KEY=os.getenv("SPACE_KEY")
ARIZE_API_KEY = os.getenv("API_KEY")

# Initialize Arize client with your space key and api key
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Define the schema for your data
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["credit_lines_outstanding", "loan_amt_outstanding", "total_debt_outstanding", "income", "years_employed", "fico_score"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)


app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = int(request.form["loan_amt_outstanding"])
        total_debt_outstanding = int(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score = int(request.form["fico_score"])

        # Assume you have actual labels available for evaluation
        actual_label = int(request.form.get("actual_label for evaluation", 1))  # Default to -1 if not provided
        #

        prediction = model.predict(
            [[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
        )
        # Log the prediction to Arize
        timestamp = pd.Timestamp.now()

        # Log the prediction to Arize
        data = {
            "prediction_id": [str(timestamp.timestamp())],  # Unique ID for each prediction
            "timestamp": [timestamp],
            "credit_lines_outstanding": [credit_lines_outstanding],
            "loan_amt_outstanding": [loan_amt_outstanding],
            "total_debt_outstanding": [total_debt_outstanding],
            "income": [income],
            "years_employed": [years_employed],
            "fico_score": [fico_score],
            "prediction_label": [int(prediction[0])],
            "actual_label": [actual_label]             
        }
        dataframe = pd.DataFrame(data)
        
        try: 
            response = arize_client.log(
                dataframe = dataframe,
                model_id="model",
                model_version="v1",
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                #features=features,
                #prediction_label = [int(prediction[0])],
                schema=schema
            )

            if response.status_code != 200:
                print(f"Failed to log data to Arize: {response.text}")
            else:
                print("Successfully logged data to Arize")
        except Exception as e:
            print(f"An error occured: {e}")
        
        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="Granting a loan to this client seems too risky!",
            )

        else:
            return render_template(
                "index.html", prediction_text="Make money money. Make money money moneeeeeey :)"
            )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
