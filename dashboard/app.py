import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

xgb = joblib.load("models/xgboost_model.pkl")
le_zone = joblib.load("models/le_zone.pkl")
le_day = joblib.load("models/le_day.pkl")

st.set_page_config(page_title="Noise Pollution Agent", layout="wide")
st.title("Smart Noise Pollution Alert Agent")

st.sidebar.header("Predict Noise Level")
zone = st.sidebar.selectbox("Zone Type", ["residential","commercial","industrial","park","transport"])
day = st.sidebar.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
traffic = st.sidebar.slider("Traffic Density", 0, 100, 50)
events = st.sidebar.slider("Nearby Events", 0, 5, 1)
temp = st.sidebar.slider("Temperature", 15, 40, 25)
wind = st.sidebar.slider("Wind Speed", 0, 30, 10)

zone_enc = le_zone.transform([zone])[0]
day_enc = le_day.transform([day])[0]
features = np.array([[hour, day_enc, zone_enc, traffic, events, temp, wind]])
prediction = xgb.predict(features)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Noise", f"{prediction:.1f} dB")
col2.metric("Zone", zone.capitalize())
col3.metric("Status", "DANGER" if prediction > 85 else "SAFE")

if prediction > 85:
    st.error(f"Warning: {prediction:.1f} dB exceeds safe threshold of 85 dB!")
else:
    st.success(f"Noise level {prediction:.1f} dB is within safe limits.")

st.divider()
st.subheader("City Noise Map")
zone_coords = {
    "residential": [26.21, 50.59],
    "commercial":  [26.22, 50.60],
    "industrial":  [26.20, 50.58],
    "park":        [26.23, 50.61],
    "transport":   [26.19, 50.57]
}
m = folium.Map(location=[26.21, 50.59], zoom_start=13)
for z, coords in zone_coords.items():
    z_enc = le_zone.transform([z])[0]
    d_enc = le_day.transform([day])[0]
    f = np.array([[hour, d_enc, z_enc, traffic, events, temp, wind]])
    pred = xgb.predict(f)[0]
    color = "red" if pred > 85 else "orange" if pred > 70 else "green"
    folium.CircleMarker(
        location=coords, radius=20,
        color=color, fill=True, fill_opacity=0.6,
        popup=f"{z}: {pred:.1f} dB"
    ).add_to(m)
st_folium(m, width=700, height=400)

st.divider()
st.subheader("Ask the Noise Agent")
question = st.text_input("Ask about noise in the city:", placeholder="Is it safe to jog near the industrial zone tonight?")
if question:
    if prediction > 85:
        answer = f"The {zone} zone at hour {hour} on {day} has a predicted noise of {prediction:.1f} dB, exceeding the 85 dB safe limit. It is NOT recommended. Consider a park zone which is much quieter."
    elif prediction > 70:
        answer = f"The {zone} zone at hour {hour} on {day} has a moderate noise level of {prediction:.1f} dB. Generally safe but may be uncomfortable for sensitive individuals."
    else:
        answer = f"The {zone} zone at hour {hour} on {day} has a low noise level of {prediction:.1f} dB, well within safe limits. It is perfectly safe for outdoor activities!"
    st.info(answer)