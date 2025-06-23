import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Monthly Climate Mapping ===
monthly_climate = {
    1: "Cloudy", 2: "Cloudy", 3: "Sunny", 4: "Sunny",
    5: "Sunny", 6: "Rainy", 7: "Rainy", 8: "Rainy",
    9: "Cloudy", 10: "Cloudy", 11: "Cloudy", 12: "Cloudy"
}

# === Sample Data for All 12 Months ===
sample_data = [
    ("2025-01-05", 16.0, 60, 4, "Cloudy"),
    ("2025-01-15", 14.0, 65, 3, "Cloudy"),
    ("2025-02-05", 18.0, 58, 5, "Cloudy"),
    ("2025-02-20", 19.0, 55, 4, "Cloudy"),
    ("2025-03-10", 25.0, 40, 6, "Sunny"),
    ("2025-03-20", 27.0, 38, 5, "Sunny"),
    ("2025-04-05", 30.0, 35, 6, "Sunny"),
    ("2025-04-25", 33.0, 30, 7, "Sunny"),
    ("2025-05-10", 38.0, 25, 8, "Sunny"),
    ("2025-05-20", 40.0, 20, 10, "Sunny"),
    ("2025-06-05", 36.0, 60, 5, "Rainy"),
    ("2025-06-20", 34.0, 65, 6, "Rainy"),
    ("2025-07-10", 30.0, 85, 4, "Rainy"),
    ("2025-07-25", 28.0, 88, 5, "Rainy"),
    ("2025-08-05", 29.0, 80, 6, "Rainy"),
    ("2025-08-20", 27.0, 85, 5, "Rainy"),
    ("2025-09-05", 30.0, 70, 6, "Cloudy"),
    ("2025-09-18", 28.0, 72, 7, "Cloudy"),
    ("2025-10-10", 26.0, 68, 6, "Cloudy"),
    ("2025-10-25", 24.0, 70, 5, "Cloudy"),
    ("2025-11-05", 22.0, 65, 4, "Cloudy"),
    ("2025-11-15", 20.0, 60, 3, "Cloudy"),
    ("2025-12-05", 18.0, 62, 4, "Cloudy"),
    ("2025-12-20", 15.0, 70, 3, "Cloudy"),
]

# === Prepare DataFrame ===
df = pd.DataFrame(sample_data, columns=["Date", "Temperature", "Humidity", "Windspeed", "Condition"])
df["Month"] = pd.to_datetime(df["Date"]).dt.month
df["AvgMonthCondition"] = df["Month"].map(monthly_climate)

# === Encode & Train ===
condition_encoder = LabelEncoder()
climate_encoder = LabelEncoder()

df["ConditionEncoded"] = condition_encoder.fit_transform(df["Condition"])
df["MonthClimateEncoded"] = climate_encoder.fit_transform(df["AvgMonthCondition"])

X = df[["Temperature", "Humidity", "Windspeed", "Month", "MonthClimateEncoded"]]
y = df["ConditionEncoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# === Streamlit App UI ===
st.set_page_config(page_title="AgriWeather Predictor", layout="centered")
st.title("🌦️ AgriWeather AI – All-Season Weather Predictor")
st.markdown("Predict weather condition using temperature, humidity, wind speed, and seasonal patterns.")

# Show Data Option
if st.checkbox("📊 Show Sample Weather Data"):
    st.dataframe(df[["Date", "Temperature", "Humidity", "Windspeed", "Month", "Condition", "AvgMonthCondition"]])

# User Input
st.subheader("🧠 Enter Today's Weather")
temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.slider("💧 Humidity (%)", 10.0, 100.0, 60.0)
wind = st.slider("🌬️ Wind Speed (km/h)", 0.0, 25.0, 5.0)
month = st.selectbox("📅 Month", list(range(1, 13)))

# Prediction
if st.button("🔮 Predict Weather Condition"):
    avg_climate = monthly_climate.get(month, "Cloudy")
    all_climates = list(set(monthly_climate.values()))
    climate_encoder.fit(all_climates)  # prevent unseen error

    climate_value = climate_encoder.transform([avg_climate])[0]
    input_scaled = scaler.transform([[temp, humidity, wind, month, climate_value]])
    pred_encoded = model.predict(input_scaled)[0]
    result = condition_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"🌤️ Predicted Weather: **{result}**")

# Show Accuracy
st.info(f"📈 Model Accuracy: **{accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%**")
