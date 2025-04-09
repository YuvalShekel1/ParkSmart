
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from googletrans import Translator

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")

st.title("Parkinson's Data Analyzer")

# העלאת קובץ JSON
uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file:
    raw_data = json.load(uploaded_file)

    # תרגום כללי של מבנה הנתונים
    translator = Translator()

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

    # תרגום מלא של הקובץ
    translated_data = translate_recursive(raw_data)

    # שמירת הקובץ כ-JSON חדש
    translated_file_path = "translated_data.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # כפתור להורדת הקובץ המתורגם
    with open(translated_file_path, "rb") as f:
        st.download_button(
            label="📥 Download Translated JSON",
            data=f,
            file_name="translated_data.json",
            mime="application/json"
        )

    # ניתוח מתוך activities בלבד
    if "activities" in translated_data:
        df = pd.json_normalize(translated_data, record_path='activities')

        st.header("Basic Pattern Detection")

        if 'intensity' in df.columns:
            st.subheader("Intensity Distribution")
            intensity_counts = df['intensity'].value_counts()
            st.bar_chart(intensity_counts)
        else:
            st.warning("Column 'intensity' not found in data.")
    else:
        st.warning("'activities' not found in the uploaded file.")
