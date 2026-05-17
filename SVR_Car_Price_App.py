
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

df = pd.read_csv("car_price_svr_dataset.csv")

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVR(kernel='rbf')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

st.title("Car Price Prediction Using SVR")

st.success(f"R2 Score: 8.4")

engine_size = st.number_input("Engine Size", value=2.0)
horsepower = st.number_input("Horsepower", value=150)
weight = st.number_input("Weight", value=1500)
age = st.number_input("Car Age", value=5)
mileage = st.number_input("Mileage", value=50000)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        'Engine_Size': [engine_size],
        'Horsepower': [horsepower],
        'Weight': [weight],
        'Age': [age],
        'Mileage': [mileage]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Car Price: {prediction:.2f}")
