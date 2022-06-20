import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

data = pd.read_csv("Churn_Modelling.csv")
result = data["Exited"]
data = data.drop(columns=["Exited", "Surname", "CustomerId", "RowNumber"])

geography_encoder = LabelEncoder()
data["Geography"] = geography_encoder.fit_transform(data["Geography"])

gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

model = Sequential()
model.add(Dense(units=5, activation="relu"))
model.add(Dense(units=5, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model.fit(X_train, y_train, batch_size=32, epochs=100)

predict = sc.transform(
    [
        [
            600,
            geography_encoder.transform(["Germany"])[0],
            gender_encoder.transform(["Male"])[0],
            40,
            3,
            60000,
            2,
            1,
            1,
            50000,
        ]
    ]
)
print(predict)
print(model.predict(predict) > 0.5)
