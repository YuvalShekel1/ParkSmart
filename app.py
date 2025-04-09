
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from googletrans import Translator
import zipfile

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")
st.title("Parkinson's Data Analyzer")

uploaded_file = st.file_uploader("Upload your ZIP file containing a JSON", type=["zip"])

def extract_json_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        for name in z.namelist():
            if name.endswith('.json'):
                with z.open(name) as f:
                    return json.load(f)
    return None

if uploaded_file:
    raw_data = extract_json_from_zip(uploaded_file)
    if raw_data is None:
        st.error("No JSON file found inside the ZIP.")
        st.stop()

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
    df_feelings = pd.json_normalize(translated_feelings)

    # שלב בחירה: בחירת סוגי type להצגה
    if "type" in df_feelings.columns:
        unique_types = df_feelings["type"].unique().tolist()
        selected_types = st.multiselect("Choose which types to display:", unique_types, default=unique_types)
        df_filtered = df_feelings[df_feelings["type"].isin(selected_types)]
    else:
        st.warning("No 'type' column found in feelings.")
        df_filtered = df_feelings

    # תצוגה
    st.subheader("Filtered Feelings Data")
    st.dataframe(df_filtered)

    st.header("Severity by Type (Filtered)")
    if 'severity' in df_filtered.columns and 'type' in df_filtered.columns:
        severity_dist = df_filtered.groupby("type")["severity"].value_counts().unstack().fillna(0)
        st.bar_chart(severity_dist)
    else:
        st.warning("Missing 'type' or 'severity' columns in feelings.")
