from flask import Flask, request

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)


def predict_codes(headline, body):
    batch = 512

    vector_path_codes = "./codes/count_vectorizer.pkl"
    model_codes = "./codes/randomforests.pkl"

    df = pd.read_csv(
        "./companies/final_updated.csv",
        usecols=["CODES", "HEADLINE", "BODY", "COMPANIES"],
        low_memory=False,
        chunksize=batch,
        nrows=1024,
    )

    for chunk in df:
        X_train, X_test, y_train, y_test = train_test_split(
            chunk[["HEADLINE", "BODY"]].apply(lambda x: " ".join(x), axis=1),
            chunk["CODES"],
            test_size=0.2,
        )

        vectorizer = joblib.load(vector_path_codes)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    vectorizer = joblib.load(vector_path_codes)
    codes_model = joblib.load(model_codes)

    X_new = vectorizer.transform([headline + " " + body])
    y_new = codes_model.predict(X_new)

    return y_new


def predict_comp(headline, body):
    batch = 512

    vector_path_comp = "./companies/count_vectorizer.pkl"
    model_comp = "./companies/randomf.pkl"

    df = pd.read_csv(
        "./companies/final_updated.csv",
        usecols=["CODES", "HEADLINE", "BODY", "COMPANIES"],
        low_memory=False,
        chunksize=batch,
        nrows=1024,
    )

    for chunk in df:
        X_train, X_test, y_train, y_test = train_test_split(
            chunk[["HEADLINE", "BODY"]].apply(lambda x: " ".join(x), axis=1),
            chunk["COMPANIES"],
            test_size=0.3,
        )

        vectorizerC = joblib.load(vector_path_comp)
        X_train_vec = vectorizerC.fit_transform(X_train)
        X_test_vec = vectorizerC.transform(X_test)

    vectorizerC = joblib.load(vector_path_comp)
    codes_model = joblib.load(model_comp)

    X_new = vectorizerC.transform([headline + " " + body])
    y_pred = codes_model.predict(X_new)

    return y_pred


@app.route("/test", methods=["POST"])
def test_endpoint():
    data = request.get_json()
    headline = data["headline"]
    body = data["body"]

    predictions_codes = predict_codes(headline, body)
    predictions_comp = predict_comp(headline, body)

    return {
        "predictions_codes": predictions_codes.tolist(),
        "predictions_comp": predictions_comp.tolist(),
    }


if __name__ == "__main__":
    app.run()
