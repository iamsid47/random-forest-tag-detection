import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


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


# Example usage
headline = "CFTC-Registered Firmsâ Adjusted Net Capital for February (Table)"
body = "By Stephen Rose April 12 (Bloomberg) -- Following is a table detailing Futures Commission Merchantsâ adjusted net capital as of the end of February. These firms act as brokers for any commodity for future delivery on or subject to the rules of any exchange. They must file monthly financial reports with the Commodity Futures Trading Commission. Adjusted net capital is the amount of regulatory capital available to meet the minimum net capital requirement as set by the CFTC. $12,411,574,774 $12,278,665,379 GOLDMAN SACHS & CO               $11,369,767,453 $11,799,253,757 $11,422,646,769 JP MORGAN SECURITIES LLC         $11,156,844,667 $10,420,866,348 $11,100,234,132 MERRILL LYNCH PIERCE FENNER & SM $11,029,181,118 $10,642,318,035 $10,761,042,594 DEUTSCHE BANK SECURITIES INC      $8,721,604,572  $8,623,920,551  $8,096,091,854"

predictions_codes = predict_codes(headline, body)
predictions_comp = predict_comp(headline, body)
print(f"Loaded Random Forest CODES: {predictions_codes}")
print(f"\n\nLoaded Random Forest COMPANIES: {predictions_comp}")
