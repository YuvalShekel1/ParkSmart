
import streamlit as st
import pandas as pd
import json
import zipfile
import os
from googletrans import Translator

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

def translate_recursive(obj, translator):
    if isinstance(obj, dict):
        return {translate_recursive(k, translator): translate_recursive(v, translator) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_recursive(i, translator) for i in obj]
    elif isinstance(obj, str):
        try:
            return translator.translate(obj, src='iw', dest='en').text
        except:
            return obj
    else:
        return obj

if uploaded_file:
    raw_data = extract_json_from_zip(uploaded_file)
    if raw_data is None:
        st.error("No JSON file found inside the ZIP.")
        st.stop()

    # תרגום מלא
    with st.spinner("Translating Hebrew text to English..."):
        translator = Translator()
        translated_data = translate_recursive(raw_data, translator)

    # שמירת הקובץ המתורגם
    translated_file_path = "translated_data.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # כפתור להורדת הקובץ
    with open(translated_file_path, "rb") as f:
        st.download_button(
            label="📥 Download Translated JSON",
            data=f,
            file_name="translated_data.json",
            mime="application/json"
        )

    # שליפת כל ה-feelings והצגת checkbox לפי type
    feelings = translated_data.get("feelings", [])
    if isinstance(feelings, list) and feelings:
        df = pd.json_normalize(feelings)
        if "type" in df.columns:
            unique_types = df["type"].dropna().unique().tolist()
            selected_types = st.multiselect("Select types to analyze:", unique_types, default=unique_types)

            df_filtered = df[df["type"].isin(selected_types)]
            st.subheader("Filtered Feelings Data")
            st.dataframe(df_filtered)

            if 'severity' in df_filtered.columns:
                st.subheader("Severity Distribution by Type")
                chart_data = df_filtered.groupby("type")["severity"].value_counts().unstack().fillna(0)
                st.bar_chart(chart_data)
            else:
                st.info("No 'severity' column found.")
        else:
            st.warning("No 'type' column found in feelings.")
    else:
        st.warning("No 'feelings' section found in the JSON.")
