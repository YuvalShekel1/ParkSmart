
import streamlit as st
import pandas as pd
import json
from googletrans import Translator
import matplotlib.pyplot as plt

# כותרת ראשית
st.title("Parkinson's Data Analyzer")

# שלב 1: העלאת קובץ JSON
uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file:
    # שלב 2: קריאה לקובץ JSON
    raw_data = json.load(uploaded_file)
    df = pd.json_normalize(raw_data)

    st.subheader("Raw Data")
    st.write(df.head())

    # שלב 3: תרגום לעברית לאנגלית
    st.subheader("Translating Hebrew to English")
    translator = Translator()

    def translate_column(col):
        return [translator.translate(str(val), src='iw', dest='en').text if isinstance(val, str) else val for val in col]

    df_translated = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df_translated[col] = translate_column(df[col])

    st.subheader("Translated Data")
    st.write(df_translated.head())

    # שלב 4: דוגמה לניתוח נתונים פשוט
    st.subheader("Basic Pattern Detection")
    if "Intensity" in df_translated.columns:
        st.write("Average Intensity by Type:")
        if "Type" in df_translated.columns:
            avg_intensity = df_translated.groupby("Type")["Intensity"].mean()
            st.bar_chart(avg_intensity)
        else:
            st.warning("Column 'Type' not found in data.")
    else:
        st.warning("Column 'Intensity' not found in data.")
