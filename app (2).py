
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from googletrans import Translator

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")
st.title("Parkinson's Data Analyzer")

uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file:
    raw_data = json.load(uploaded_file)
    translator = Translator()

    # תרגום רק של חלק ה-feelings מ-עברית לאנגלית
    def translate_feelings_section(feelings):
        def translate_recursive(data):
            if isinstance(data, dict):
                return {translate_recursive(k): translate_recursive(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [translate_recursive(item) for item in data]
            elif isinstance(data, str):
                try:
                    translated = translator.translate(data, src='iw', dest='en').text
                    return translated
                except:
                    return data
            else:
                return data
        return translate_recursive(feelings)

    translated_feelings = translate_feelings_section(raw_data.get("feelings", []))

    # עדכון הנתונים המתורגמים במבנה המקורי
    translated_data = raw_data.copy()
    translated_data["feelings"] = translated_feelings

    # שמירת JSON מתורגם חדש
    translated_file_path = "translated_feelings_to_english.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # כפתור להורדה
    with open(translated_file_path, "rb") as f:
        st.download_button(
            label="📥 Download JSON (Feelings translated to English)",
            data=f,
            file_name="translated_feelings_to_english.json",
            mime="application/json"
        )

    # הצגת טבלה עם הנתונים המתורגמים
    df_feelings = pd.json_normalize(translated_feelings)
    st.subheader("Translated Feelings (to English)")
    st.dataframe(df_feelings)

    # גרף התפלגות severity לפי type
    st.header("Severity by Type in Feelings")
    if 'severity' in df_feelings.columns and 'type' in df_feelings.columns:
        severity_dist = df_feelings.groupby("type")["severity"].value_counts().unstack().fillna(0)
        st.bar_chart(severity_dist)
    else:
        st.warning("Missing 'type' or 'severity' columns in feelings.")
