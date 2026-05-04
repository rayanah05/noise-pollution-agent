import streamlit as st
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(page_title="Noise Pollution Agent", layout="wide", page_icon="🔊")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e0ff;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.2rem;
    letter-spacing: -1px;
}

.subtitle {
    text-align: center;
    color: #7090b0;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,255,0.08));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.metric-label {
    font-size: 0.75rem;
    color: #7090b0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
}

.metric-value.danger { color: #ff4444; }
.metric-value.safe { color: #00ff88; }
.metric-value.zone { color: #c084fc; }

.alert-danger {
    background: linear-gradient(135deg, rgba(255,68,68,0.15), rgba(255,68,68,0.05));
    border: 1px solid rgba(255,68,68,0.4);
    border-left: 4px solid #ff4444;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #ff9999;
    font-weight: 500;
    margin: 1rem 0;
}

.alert-safe {
    background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,255,136,0.03));
    border: 1px solid rgba(0,255,136,0.3);
    border-left: 4px solid #00ff88;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #80ffbb;
    font-weight: 500;
    margin: 1rem 0;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.agent-response {
    background: linear-gradient(135deg, rgba(123,47,255,0.12), rgba(0,212,255,0.06));
    border: 1px solid rgba(123,47,255,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    color: #d0d0ff;
    font-size: 1rem;
    line-height: 1.7;
    margin-top: 0.8rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(0,212,255,0.1);
    margin: 1.5rem 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a, #0a1020) !important;
    border-right: 1px solid rgba(0,212,255,0.1);
}

[data-testid="stSidebar"] label {
    color: #a0b4c8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #7b2fff) !important;
}

.sidebar-title {
    color: #00d4ff;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,212,255,0.2);
}
</style>
""", unsafe_allow_html=True)

xgb = joblib.load("models/xgboost_model.pkl")
le_zone = joblib.load("models/le_zone.pkl")
le_day = joblib.load("models/le_day.pkl")

st.markdown('<div class="main-title">🔊 Smart Noise Pollution Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Urban Sound Intelligence • Bahrain City Monitor</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Control Panel</div>', unsafe_allow_html=True)
    zone = st.selectbox("Zone Type", ["residential","commercial","industrial","park","transport"])
    day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    hour = st.slider("Hour of Day", 0, 23, 12, format="%d:00")
    traffic = st.slider("Traffic Density", 0, 100, 50)
    events = st.slider("Nearby Events", 0, 5, 1)
    temp = st.slider("Temperature (°C)", 15, 40, 25)
    wind = st.slider("Wind Speed (km/h)", 0, 30, 10)

zone_enc = le_zone.transform([zone])[0]
day_enc = le_day.transform([day])[0]
features = np.array([[hour, day_enc, zone_enc, traffic, events, temp, wind]])
prediction = xgb.predict(features)[0]
is_danger = prediction > 85

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Noise</div><div class="metric-value {"danger" if is_danger else "safe"}">{prediction:.1f}</div><div style="color:#7090b0;font-size:0.8rem">decibels</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Zone</div><div class="metric-value zone">{zone.upper()}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value" style="color:#fbbf24">{hour:02d}:00</div><div style="color:#7090b0;font-size:0.8rem">{day}</div></div>', unsafe_allow_html=True)
with col4:
    status = "DANGER" if is_danger else "SAFE"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Status</div><div class="metric-value {"danger" if is_danger else "safe"}">{status}</div></div>', unsafe_allow_html=True)

if is_danger:
    st.markdown(f'<div class="alert-danger">⚠ Warning: Predicted noise level of {prediction:.1f} dB exceeds the WHO safe threshold of 85 dB. Prolonged exposure may cause hearing damage.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-safe">✓ Noise level {prediction:.1f} dB is within safe limits. This zone is suitable for outdoor activities.</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

col_gauge, col_map = st.columns([1, 2])

with col_gauge:
    st.markdown('<div class="section-header">📊 Noise Gauge</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        delta={'reference': 85, 'valueformat': '.1f'},
        title={'text': "dB Level", 'font': {'color': '#a0b4c8', 'size': 14}},
        number={'font': {'color': '#00d4ff', 'size': 36}, 'suffix': ' dB'},
        gauge={
            'axis': {'range': [30, 110], 'tickcolor': '#7090b0', 'tickfont': {'color': '#7090b0'}},
            'bar': {'color': '#ff4444' if is_danger else '#00d4ff', 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [30, 70], 'color': 'rgba(0,255,136,0.15)'},
                {'range': [70, 85], 'color': 'rgba(255,165,0,0.15)'},
                {'range': [85, 110], 'color': 'rgba(255,68,68,0.15)'}
            ],
            'threshold': {'line': {'color': '#ff4444', 'width': 2}, 'thickness': 0.8, 'value': 85}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#a0b4c8'},
        height=250,
        margin=dict(l=20, r=20, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📈 Hourly Trend</div>', unsafe_allow_html=True)
    hours = list(range(24))
    preds = []
    for h in hours:
        f = np.array([[h, day_enc, zone_enc, traffic, events, temp, wind]])
        preds.append(xgb.predict(f)[0])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hours, y=preds,
        mode='lines+markers',
        line=dict(color='#00d4ff', width=2),
        marker=dict(color=['#ff4444' if p > 85 else '#00d4ff' for p in preds], size=6),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.05)'
    ))
    fig2.add_hline(y=85, line_dash="dash", line_color="#ff4444", annotation_text="85dB limit", annotation_font_color="#ff4444")
    fig2.add_vline(x=hour, line_dash="dot", line_color="#fbbf24")
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Hour', color='#7090b0', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='dB', color='#7090b0', gridcolor='rgba(255,255,255,0.05)'),
        height=200,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_map:
    st.markdown('<div class="section-header">🗺 City Noise Map</div>', unsafe_allow_html=True)
    zone_coords = {
        "residential": [26.21, 50.59],
        "commercial":  [26.22, 50.60],
        "industrial":  [26.20, 50.58],
        "park":        [26.23, 50.61],
        "transport":   [26.19, 50.57]
    }
    m = folium.Map(location=[26.21, 50.59], zoom_start=13,
                   tiles='CartoDB dark_matter')
    for z, coords in zone_coords.items():
        z_enc = le_zone.transform([z])[0]
        f = np.array([[hour, day_enc, z_enc, traffic, events, temp, wind]])
        pred = xgb.predict(f)[0]
        color = "#ff4444" if pred > 85 else "#ffa500" if pred > 70 else "#00ff88"
        folium.CircleMarker(
            location=coords, radius=25,
            color=color, fill=True, fill_opacity=0.5,
            weight=2,
            popup=folium.Popup(f"<b>{z.upper()}</b><br>{pred:.1f} dB<br>{'⚠ DANGER' if pred > 85 else '✓ SAFE'}", max_width=150),
            tooltip=f"{z.capitalize()}: {pred:.1f} dB"
        ).add_to(m)
        folium.Marker(
            location=coords,
            icon=folium.DivIcon(html=f'<div style="color:white;font-size:10px;font-weight:bold;text-align:center;white-space:nowrap;margin-top:-8px">{pred:.0f}dB</div>')
        ).add_to(m)
    st_folium(m, width=None, height=460)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🤖 Ask the Noise Agent</div>', unsafe_allow_html=True)

question = st.text_input("", placeholder="e.g. Is it safe to jog near the industrial zone tonight?")
if question:
    if prediction > 85:
        answer = f"🚨 The {zone} zone at {hour:02d}:00 on {day} has a predicted noise level of {prediction:.1f} dB, which significantly exceeds the WHO safe threshold of 85 dB. I strongly advise against prolonged exposure in this area. Consider visiting the park zone instead, which is currently much quieter and safer for outdoor activities."
    elif prediction > 70:
        answer = f"⚠ The {zone} zone at {hour:02d}:00 on {day} shows a moderate noise level of {prediction:.1f} dB. While below the 85 dB danger threshold, sensitive individuals such as children or the elderly may find it uncomfortable. Short visits are fine but avoid prolonged stay."
    else:
        answer = f"✅ Great news! The {zone} zone at {hour:02d}:00 on {day} has a low noise level of {prediction:.1f} dB, well within safe limits. It is perfectly suitable for jogging, outdoor dining, or any recreational activity. Enjoy your time!"
    st.markdown(f'<div class="agent-response">{answer}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#3a5a7a;font-size:0.75rem;padding:0.5rem">DSAI 465 • Smart Noise Pollution Alert Agent • Bahrain</div>', unsafe_allow_html=True)
