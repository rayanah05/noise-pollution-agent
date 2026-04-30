code = open('dashboard/app.py', 'r', encoding='utf-8').read()

old = '''    with st.spinner("Thinking..."):
        context = f"You are a noise pollution assistant. Current prediction: {zone} zone at hour {hour} on {day} = {prediction:.1f} dB. Safe threshold is 85 dB. Answer in 2-3 sentences."
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(context + "\\n\\nUser question: " + question)
        st.info(response.text)'''

new = '''    with st.spinner("Thinking..."):
        if prediction > 85:
            answer = f"The {zone} zone at hour {hour} on {day} has a predicted noise level of {prediction:.1f} dB, which exceeds the safe threshold of 85 dB. It is NOT recommended to be in this area. Consider visiting a park zone which tends to be quieter."
        elif prediction > 70:
            answer = f"The {zone} zone at hour {hour} on {day} has a moderate noise level of {prediction:.1f} dB. It is generally safe but may be slightly uncomfortable for sensitive individuals."
        else:
            answer = f"The {zone} zone at hour {hour} on {day} has a low noise level of {prediction:.1f} dB, well within safe limits. It is perfectly safe for outdoor activities."
        st.info(answer)'''

code = code.replace(old, new)
open('dashboard/app.py', 'w', encoding='utf-8').write(code)
print('Fixed!')