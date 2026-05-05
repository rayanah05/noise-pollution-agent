code = open('dashboard/app.py', 'r', encoding='utf-8').read()

old = '''    if question:
        if prediction > 85:
            answer = f"The {zone} zone at {hour:02d}:00 on {day} has a predicted noise level of {prediction:.1f} dB, which exceeds the WHO safe threshold of 85 dB. Prolonged exposure may cause hearing damage. I strongly advise against prolonged exposure in this area. The industrial zone typically has the highest noise levels due to machinery and heavy vehicles. Consider visiting the park zone instead, which currently shows much lower noise levels and is ideal for outdoor activities."
        elif prediction > 70:
            answer = f"The {zone} zone at {hour:02d}:00 on {day} shows a moderate noise level of {prediction:.1f} dB. While this is below the 85 dB danger threshold, it may still be uncomfortable for sensitive individuals such as children or the elderly. Short visits are generally fine, but I would recommend avoiding prolonged stays during peak hours."
        else:
            answer = f"Great news! The {zone} zone at {hour:02d}:00 on {day} has a low noise level of {prediction:.1f} dB, well within safe limits. This zone is perfectly suitable for jogging, outdoor dining, studying, or any recreational activity. The noise level is comparable to a normal conversation."
        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()'''

new = '''    if question:
        q_lower = question.lower()
        detected_zone = zone
        if "industrial" in q_lower:
            detected_zone = "industrial"
        elif "commercial" in q_lower:
            detected_zone = "commercial"
        elif "residential" in q_lower:
            detected_zone = "residential"
        elif "park" in q_lower:
            detected_zone = "park"
        elif "transport" in q_lower:
            detected_zone = "transport"
        dz_enc = le_zone.transform([detected_zone])[0]
        q_pred = round(xgb.predict(np.array([[hour, day_enc, dz_enc, traffic, events, temp, wind]]))[0], 1)
        if q_pred > 85:
            answer = f"The {detected_zone} zone at {hour:02d}:00 on {day} has a predicted noise of {q_pred:.1f} dB, exceeding the 85 dB safe limit. Prolonged exposure may cause hearing damage. Consider visiting the park zone instead which is much quieter."
        elif q_pred > 70:
            answer = f"The {detected_zone} zone at {hour:02d}:00 on {day} shows a moderate noise level of {q_pred:.1f} dB. Generally safe for short visits but may be uncomfortable for sensitive individuals like children or the elderly."
        else:
            answer = f"The {detected_zone} zone at {hour:02d}:00 on {day} has a low noise level of {q_pred:.1f} dB, well within safe limits. It is perfectly suitable for any outdoor activity!"
        st.session_state.history.append({"q": question, "a": answer})
        st.rerun()'''

if old in code:
    code = code.replace(old, new)
    open('dashboard/app.py', 'w', encoding='utf-8').write(code)
    print('Fixed!')
else:
    print('Text not found - will patch differently')
    code = code.replace("        if prediction > 85:\n            answer = f\"The {zone}", "        q_lower = question.lower()\n        detected_zone = zone\n        if 'industrial' in q_lower:\n            detected_zone = 'industrial'\n        elif 'commercial' in q_lower:\n            detected_zone = 'commercial'\n        elif 'residential' in q_lower:\n            detected_zone = 'residential'\n        elif 'park' in q_lower:\n            detected_zone = 'park'\n        elif 'transport' in q_lower:\n            detected_zone = 'transport'\n        dz_enc = le_zone.transform([detected_zone])[0]\n        q_pred = round(xgb.predict(np.array([[hour, day_enc, dz_enc, traffic, events, temp, wind]]))[0], 1)\n        if q_pred > 85:\n            answer = f\"The {detected_zone}")
    open('dashboard/app.py', 'w', encoding='utf-8').write(code)
    print('Patched!')