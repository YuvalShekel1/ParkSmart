
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from googletrans import Translator
import zipfile
import io

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")
st.title("Parkinson's Data Analyzer")

uploaded_file = st.file_uploader("Upload your JSON or ZIP file", type=["json", "zip"])

def extract_json_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        file_names = z.namelist()
        for name in file_names:
            if name.endswith('.json'):
                with z.open(name) as f:
                    return json.load(f)
    return None

if uploaded_file:
    if uploaded_file.name.endswith('.zip'):
        raw_data = extract_json_from_zip(uploaded_file)
        if raw_data is None:
            st.error("No JSON file found inside the ZIP.")
            st.stop()
    else:
        raw_data = json.load(uploaded_file)

    translator = Translator()

    def translate_feelings_section(feelings):
        def translate_recursive(data):
            if isinstance(data, dict):
                return {translate_recursive(k): translate_recursive(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [translate_recursive(item) for item in data]
            elif isinstance(data, str):
                try:
                    return translator.translate(data, src='iw', dest='en').text
                except:
                    return data
            else:
                return data
        return translate_recursive(feelings)

    translated_feelings = translate_feelings_section(raw_data.get("feelings", []))
    translated_data = raw_data.copy()
    translated_data["feelings"] = translated_feelings

    translated_file_path = "translated_feelings_to_english.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    with open(translated_file_path, "rb") as f:
        st.download_button(
            label="📥 Download JSON (Feelings translated to English)",
            data=f,
            file_name="translated_feelings_to_english.json",
            mime="application/json"
        )

    df_feelings = pd.json_normalize(translated_feelings)
    st.subheader("Translated Feelings (to English)")
    st.dataframe(df_feelings)

    st.header("Severity by Type in Feelings")
    if 'severity' in df_feelings.columns and 'type' in df_feelings.columns:
        severity_dist = df_feelings.groupby("type")["severity"].value_counts().unstack().fillna(0)
        st.bar_chart(severity_dist)
    else:
        st.warning("Missing 'type' or 'severity' columns in feelings.")
