import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Noise Pollution Alert Agent", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f8f9fc; }

.page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    color: white;
    padding: 1.8rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.page-header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.page-header p { font-size: 0.8rem; opacity: 0.7; margin: 0.3rem 0 0; letter-spacing: 1px; text-transform: uppercase; }

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-label {
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #1e3a5f;
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}
.metric-value.danger { color: #dc2626; }
.metric-value.safe { color: #16a34a; }
.metric-value.zone { color: #2d6a9f; font-size: 1.4rem; }
.metric-value.time { color: #d97706; }

.alert-danger {
    background: #fef2f2; border: 1px solid #fecaca;
    border-left: 4px solid #dc2626; border-radius: 8px;
    padding: 0.9rem 1.2rem; color: #991b1b;
    font-weight: 500; margin: 1rem 0; font-size: 0.9rem;
}
.alert-safe {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 8px;
    padding: 0.9rem 1.2rem; color: #166534;
    font-weight: 500; margin: 1rem 0; font-size: 0.9rem;
}
.alert-moderate {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #d97706; border-radius: 8px;
    padding: 0.9rem 1.2rem; color: #92400e;
    font-weight: 500; margin: 1rem 0; font-size: 0.9rem;
}

.section-title {
    font-size: 0.85rem; font-weight: 600; color: #1e3a5f;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 1.2rem 0 0.8rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

.chat-question {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px 10px 10px 2px; padding: 0.7rem 1rem;
    color: #1e40af; font-size: 0.9rem; margin: 0.5rem 0;
}
.chat-answer {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px 10px 2px 10px; padding: 0.7rem 1rem;
    color: #334155; font-size: 0.9rem; margin: 0.5rem 0 1rem;
    line-height: 1.6; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.info-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.2rem 1.5rem;
    margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

[data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] label { color: #475569 !important; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
.sidebar-header { background: #1e3a5f; color: white; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }
</style>
""", unsafe_allow_html=True)

xgb = joblib.load("models/xgboost_model.pkl")
le_zone = joblib.load("models/le_zone.pkl")
le_day = joblib.load("models/le_day.pkl")

def predict_noise(h, d_enc, z, t, e, tmp, w):
    z_enc = le_zone.transform([z])[0]
    f = np.array([[h, d_enc, z_enc, t, e, tmp, w]])
    return round(float(xgb.predict(f)[0]), 1)

def detect_zone(question, default_zone):
    q = question.lower()
    # Direct zone mentions
    if "industrial" in q: return "industrial"
    if "commercial" in q: return "commercial"
    if "residential" in q: return "residential"
    if "park" in q or "picnic" in q or "jog" in q or "bike" in q or "walk" in q or "outdoor" in q: return "park"
    if "transport" in q or "highway" in q or "road" in q or "traffic" in q: return "transport"
    if "shop" in q or "market" in q or "restaurant" in q or "cafe" in q or "coffee" in q or "drink" in q: return "commercial"
    if "factory" in q or "work" in q or "industrial" in q or "machine" in q: return "industrial"
    if "home" in q or "house" in q or "neighborhood" in q or "sleep" in q or "study" in q or "kids" in q or "children" in q: return "residential"
    return default_zone

def generate_answer(question, detected_zone, q_pred, hour, day, all_zone_preds):
    q = question.lower()
    status = "dangerous" if q_pred > 85 else "moderate" if q_pred > 70 else "safe"
    
    # Comparison questions
    if "safest" in q or "quietest" in q or "which zone" in q or "best zone" in q:
        sorted_zones = sorted(all_zone_preds.items(), key=lambda x: x[1])
        safest = sorted_zones[0]
        loudest = sorted_zones[-1]
        return f"Based on current conditions at {hour:02d}:00 on {day}, the quietest zone is {safest[0]} at {safest[1]} dB, while the loudest is {loudest[0]} at {loudest[1]} dB. All zone noise levels: " + ", ".join([f"{z}: {p} dB" for z, p in sorted_zones]) + ". The park zone is generally the most suitable for quiet outdoor activities."
    
    if "quieter" in q or "noisier" in q or "compare" in q or "vs" in q or "than" in q:
        sorted_zones = sorted(all_zone_preds.items(), key=lambda x: x[1])
        return f"At {hour:02d}:00 on {day}, noise levels across all zones are: " + ", ".join([f"{z}: {p} dB" for z, p in sorted_zones]) + f". The {detected_zone} zone currently measures {q_pred} dB which is {status}."

    # Activity-specific answers
    if "jog" in q or "run" in q or "exercise" in q or "bike" in q or "walk" in q:
        if q_pred > 85:
            return f"Jogging or exercising in the {detected_zone} zone at {hour:02d}:00 on {day} is NOT recommended. The noise level of {q_pred} dB exceeds the WHO safe limit of 85 dB, which can cause hearing damage during prolonged physical activity. Try the park zone instead."
        elif q_pred > 70:
            return f"Jogging in the {detected_zone} zone at {hour:02d}:00 on {day} is acceptable but not ideal. The noise level is {q_pred} dB — below the danger threshold but still moderately loud. Consider wearing ear protection or visiting during quieter hours."
        else:
            return f"The {detected_zone} zone at {hour:02d}:00 on {day} is great for jogging or outdoor exercise! The noise level is only {q_pred} dB, well within safe limits. Enjoy your workout!"

    if "picnic" in q or "kids" in q or "children" in q or "family" in q:
        if q_pred > 85:
            return f"A picnic or family outing in the {detected_zone} zone at {hour:02d}:00 on {day} is NOT advisable. At {q_pred} dB, the noise exceeds safe levels especially for children. The park zone is a much better option for families."
        elif q_pred > 70:
            return f"The {detected_zone} zone at {hour:02d}:00 on {day} has a moderate noise level of {q_pred} dB. It is acceptable for a short family visit but may be uncomfortable for young children or elderly. Consider an earlier time for quieter conditions."
        else:
            return f"The {detected_zone} zone at {hour:02d}:00 on {day} is perfect for a picnic or family outing! At only {q_pred} dB, the noise level is very comfortable and safe for children and elderly alike."

    if "study" in q or "work" in q or "concentrate" in q or "focus" in q:
        if q_pred > 85:
            return f"Studying or working in the {detected_zone} zone at {hour:02d}:00 is not recommended — at {q_pred} dB, the noise level is dangerously high and would severely impact concentration and hearing health."
        elif q_pred > 70:
            return f"The {detected_zone} zone at {hour:02d}:00 has a moderate noise level of {q_pred} dB. Studying here would be challenging without noise-cancelling headphones. A quieter time or the residential zone may be better."
        else:
            return f"The {detected_zone} zone at {hour:02d}:00 is suitable for studying or focused work at {q_pred} dB. The ambient noise is low enough for comfortable concentration."

    if "sleep" in q or "rest" in q or "nap" in q:
        if q_pred > 70:
            return f"Sleeping near the {detected_zone} zone at {hour:02d}:00 on {day} would be difficult. At {q_pred} dB, the noise level is too high for restful sleep. WHO recommends below 40 dB for sleeping. Consider a different time or zone."
        else:
            return f"The {detected_zone} zone at {hour:02d}:00 on {day} has a relatively low noise level of {q_pred} dB, making it more suitable for rest compared to other zones. However, for ideal sleep conditions, noise levels below 40 dB are recommended."

    # Default answer
    if q_pred > 85:
        return f"The {detected_zone} zone at {hour:02d}:00 on {day} has a predicted noise level of {q_pred} dB, which exceeds the WHO safe threshold of 85 dB. Prolonged exposure may cause hearing damage. It is strongly advised to avoid staying in this area for long. The park zone is currently much quieter and safer."
    elif q_pred > 70:
        return f"The {detected_zone} zone at {hour:02d}:00 on {day} has a moderate noise level of {q_pred} dB. While below the 85 dB danger threshold, it may be uncomfortable for sensitive individuals. Short visits are generally fine."
    else:
        return f"The {detected_zone} zone at {hour:02d}:00 on {day} has a low noise level of {q_pred} dB, well within safe limits. It is perfectly suitable for any outdoor activity including jogging, dining, or recreational use."

def get_status(pred):
    if pred > 85: return "DANGER", "danger"
    if pred > 70: return "MODERATE", "moderate"
    return "SAFE", "safe"

# Header
st.markdown('<div class="page-header"><h1>Smart Noise Pollution Alert Agent</h1><p>AI-Powered Urban Sound Intelligence — Bahrain City Monitor</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">Control Panel</div>', unsafe_allow_html=True)
    zone = st.selectbox("Zone Type", ["residential","commercial","industrial","park","transport"])
    day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    hour = st.slider("Hour of Day", 0, 23, 12, format="%d:00")
    traffic = st.slider("Traffic Density", 0, 100, 50)
    events = st.slider("Nearby Events", 0, 5, 1)
    temp = st.slider("Temperature (C)", 15, 40, 25)
    wind = st.slider("Wind Speed (km/h)", 0, 30, 10)


day_enc = le_day.transform([day])[0]
prediction = predict_noise(hour, day_enc, zone, traffic, events, temp, wind)
is_danger = prediction > 85
is_moderate = 70 < prediction <= 85
status_text, status_class = get_status(prediction)

# Metric cards — equal height fixed
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Noise</div><div class="metric-value {status_class}">{prediction:.1f} dB</div><div class="metric-sub">decibels</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Zone</div><div class="metric-value zone">{zone.capitalize()}</div><div class="metric-sub">selected zone</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value time">{hour:02d}:00</div><div class="metric-sub">{day}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value {status_class}">{status_text}</div><div class="metric-sub">WHO 85 dB limit</div></div>', unsafe_allow_html=True)

# Alert banner
if is_danger:
    st.markdown(f'<div class="alert-danger">Warning: {prediction:.1f} dB exceeds the WHO safe threshold of 85 dB. Prolonged exposure may cause hearing damage.</div>', unsafe_allow_html=True)
elif is_moderate:
    st.markdown(f'<div class="alert-moderate">Moderate noise level of {prediction:.1f} dB detected. Generally safe but may be uncomfortable for sensitive individuals.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-safe">Noise level {prediction:.1f} dB is within safe limits. This zone is suitable for outdoor activities.</div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Dashboard", "City Map", "AI Agent"])

with tab1:
    col_gauge, col_trend = st.columns([1, 2])
    with col_gauge:
        st.markdown('<div class="section-title">Noise Level Gauge</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={'reference': 85, 'valueformat': '.1f'},
            title={'text': "Current dB Level", 'font': {'color': '#1e3a5f', 'size': 13}},
            number={'font': {'color': '#dc2626' if is_danger else '#16a34a', 'size': 40}, 'suffix': ' dB'},
            gauge={
                'axis': {'range': [30, 110], 'tickcolor': '#94a3b8', 'tickfont': {'color': '#64748b', 'size': 11}},
                'bar': {'color': '#dc2626' if is_danger else '#2d6a9f', 'thickness': 0.25},
                'bgcolor': 'white', 'borderwidth': 1, 'bordercolor': '#e2e8f0',
                'steps': [{'range': [30,70], 'color': '#f0fdf4'}, {'range': [70,85], 'color': '#fefce8'}, {'range': [85,110], 'color': '#fef2f2'}],
                'threshold': {'line': {'color': '#dc2626', 'width': 2}, 'thickness': 0.8, 'value': 85}
            }
        ))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', height=270, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Zone Comparison</div>', unsafe_allow_html=True)
        zones_list = ["residential","commercial","industrial","park","transport"]
        zone_preds = [predict_noise(hour, day_enc, z, traffic, events, temp, wind) for z in zones_list]
        colors = ['#dc2626' if p > 85 else '#d97706' if p > 70 else '#16a34a' for p in zone_preds]
        fig3 = go.Figure(go.Bar(x=zones_list, y=zone_preds, marker_color=colors, text=zone_preds, textposition='outside'))
        fig3.add_hline(y=85, line_dash="dash", line_color="#dc2626", annotation_text="85 dB limit")
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='white', height=220,
                           xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9', range=[0,120]),
                           margin=dict(l=10,r=10,t=10,b=10), font=dict(color='#475569', size=11))
        st.plotly_chart(fig3, use_container_width=True)

    with col_trend:
        st.markdown('<div class="section-title">24-Hour Noise Trend</div>', unsafe_allow_html=True)
        hours_list = list(range(24))
        hourly_preds = [predict_noise(h, day_enc, zone, traffic, events, temp, wind) for h in hours_list]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hours_list, y=hourly_preds, mode='lines+markers',
            line=dict(color='#2d6a9f', width=2.5),
            marker=dict(color=['#dc2626' if p > 85 else '#d97706' if p > 70 else '#16a34a' for p in hourly_preds], size=7),
            fill='tozeroy', fillcolor='rgba(45,106,159,0.06)'
        ))
        fig2.add_hline(y=85, line_dash="dash", line_color="#dc2626", annotation_text="WHO 85 dB limit")
        fig2.add_vline(x=hour, line_dash="dot", line_color="#d97706", annotation_text=f"{hour:02d}:00")
        fig2.update_layout(
            paper_bgcolor='white', plot_bgcolor='white', height=280,
            xaxis=dict(title='Hour of Day', gridcolor='#f1f5f9', tickmode='linear', dtick=2, color='#64748b'),
            yaxis=dict(title='Noise Level (dB)', gridcolor='#f1f5f9', range=[30,115], color='#64748b'),
            margin=dict(l=10,r=10,t=10,b=40), showlegend=False, font=dict(color='#475569')
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Key Statistics</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Min (24h)", f"{min(hourly_preds):.1f} dB")
        s2.metric("Max (24h)", f"{max(hourly_preds):.1f} dB")
        s3.metric("Average", f"{sum(hourly_preds)/len(hourly_preds):.1f} dB")
        s4.metric("Danger Hours", f"{sum(1 for p in hourly_preds if p > 85)}/24")

with tab2:
    st.markdown('<div class="section-title">Bahrain City Noise Map</div>', unsafe_allow_html=True)
    st.caption("Green: Safe below 70 dB   |   Orange: Moderate 70-85 dB   |   Red: Danger above 85 dB   |   Click circles for details.")
    zone_coords = {"residential":[26.2154,50.5861],"commercial":[26.2285,50.5860],"industrial":[26.1921,50.5131],"park":[26.2455,50.6015],"transport":[26.2076,50.5479]}
    m = folium.Map(location=[26.22,50.57], zoom_start=12, tiles='CartoDB positron')
    for z, coords in zone_coords.items():
        pred = predict_noise(hour, day_enc, z, traffic, events, temp, wind)
        color = "#dc2626" if pred > 85 else "#d97706" if pred > 70 else "#16a34a"
        st_label = "DANGER" if pred > 85 else "MODERATE" if pred > 70 else "SAFE"
        folium.CircleMarker(location=coords, radius=30, color=color, fill=True, fill_opacity=0.35, weight=2,
            popup=folium.Popup(f"<b>{z.upper()}</b><br>Noise: {pred} dB<br>Status: {st_label}<br>Time: {hour:02d}:00 {day}", max_width=200),
            tooltip=f"{z.capitalize()}: {pred} dB").add_to(m)
        folium.Marker(location=coords,
            icon=folium.DivIcon(html=f'<div style="color:white;font-size:11px;font-weight:700;text-align:center;background:{color};padding:3px 6px;border-radius:4px;white-space:nowrap">{pred}dB</div>', icon_size=(60,25), icon_anchor=(30,12))
        ).add_to(m)
    st_folium(m, width=None, height=520)

with tab3:
    st.markdown('<div class="section-title">Intelligent Noise Agent</div>', unsafe_allow_html=True)
    st.caption("Ask about any zone — the agent detects the zone from your question automatically.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for chat in st.session_state.history:
        st.markdown(f'<div class="chat-question">{chat["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-answer">{chat["a"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask about noise levels, safety, or recommendations...")
    if question:
        detected_zone = detect_zone(question, zone)
        q_pred = predict_noise(hour, day_enc, detected_zone, traffic, events, temp, wind)
        all_zone_preds = {z: predict_noise(hour, day_enc, z, traffic, events, temp, wind) for z in ["residential","commercial","industrial","park","transport"]}
        answer = generate_answer(question, detected_zone, q_pred, hour, day, all_zone_preds)
        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()

    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.75rem;padding:0.5rem">Smart Noise Pollution Alert Agent — American University of Bahrain</div>', unsafe_allow_html=True)