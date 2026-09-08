import joblib


def predict(data):
    model = joblib.load("models/iris_model.pkl")
    prediction = model.predict([data])

    return prediction[0]


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict(sample)

    print("Predicted Class:", result)