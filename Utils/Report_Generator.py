import json
import numpy as np
import requests
import streamlit as st

def generate_tumor_report(features, URL=None, model=None, predicted_label=None, lang='en'):
    """
    Generate a detailed AI tumor report using gpt-oss:20b from your Ollama server.
    Streams the model output live into Streamlit UI.
    """

    # ----------- Convert features to JSON-safe format -----------
    serializable_features = {}
    for k, v in features.items():
        if isinstance(v, (np.ndarray, list, tuple)):
            serializable_features[k] = [float(x) for x in np.array(v).flatten()]
        elif isinstance(v, (np.generic,)):
            serializable_features[k] = float(v)
        else:
            serializable_features[k] = v

    # ----------- Prompt template -----------
    prompt = f"""
You are a professional medical imaging analyst AI specialized in MRI-based brain tumor analysis.
Generate a concise, structured tumor report based on the extracted MRI features and classification result.

Tumor Type Prediction: {predicted_label}

Quantitative MRI Features (normalized units):
{json.dumps(serializable_features, indent=2)}

Instructions:
0. Report line by line
1. Write a brief, clinically interpretable summary (maximum 5 sentences) describing:
   - Tumor size, shape, and texture.
   - Notable feature abnormalities (if any).
2. Reference key numerical values naturally in the description.
3. End with a short advice line for clinicians or radiologists.
4. Maintain a factual, professional tone (no repetition).
5. Write in { 'Vietnamese' if lang == 'vi' else 'English' }.
6. Keep the report under 200 words.
"""

    # ----------- Stream setup -----------
    url = URL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "temperature": 0.1
    }

    # ----------- Live Stream Output -----------
    message_placeholder = st.empty()
    report_text = ""

    with st.spinner("Generating advanced tumor analysis report (please wait 30–60s)..."):
        try:
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        chunk = data.get("response", "")
                        report_text += chunk
                        # Live update (typing effect)
                        message_placeholder.markdown(report_text)
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to model server: {e}")
            return "Error: Unable to connect to the inference server."

    st.warning("⚠️ **Disclaimer:** This report is AI-generated for research and educational purposes only and should not be interpreted as medical advice.")
    st.success("Tumor report successfully generated.")
    return report_text
