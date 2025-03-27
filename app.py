
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from googletrans import Translator

# כותרת ראשית
st.title("Parkinson's Data Analyzer")

# שלב 1: העלאת קובץ JSON
uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file:
    # שלב 2: קריאה לקובץ JSON מתוך activities
    raw_data = json.load(uploaded_file)

    # נורמליזציה רק לרשימת הפעילויות
    if "activities" in raw_data:
        df = pd.json_normalize(raw_data, record_path='activities')
    else:
        st.error("No 'activities' key found in JSON.")
        st.stop()

    # תרגום עמודות מאחורי הקלעים
    translator = Translator()

    def translate_column(col):
        return [translator.translate(str(val), src='iw', dest='en').text if isinstance(val, str) else val for val in col]

    df_translated = df.copy()
    for col in df_translated.columns:
        if df_translated[col].dtype == object:
            try:
                df_translated[col] = translate_column(df_translated[col])
            except:
                pass  # מדלג על שגיאות בתרגום

    # שלב 3: זיהוי דפוסים
    st.header("Basic Pattern Detection")

    if 'intensity' in df_translated.columns:
        st.subheader("Intensity Distribution")
        intensity_counts = df_translated['intensity'].value_counts()
        st.bar_chart(intensity_counts)
    else:
        st.warning("Column 'intensity' not found in data.")
