
import streamlit as st
import pandas as pd
import json
from googletrans import Translator

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")
st.title("Parkinson's Data Analyzer")

# שלב 1: העלאת קובץ JSON
uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])

if uploaded_file:
    # שלב 2: קריאה לקובץ JSON
    raw_data = json.load(uploaded_file)
    df = pd.json_normalize(raw_data)

    # שלב 3: תרגום מאחורי הקלעים
    translator = Translator()

    def translate_column(col):
        return [translator.translate(str(val), src='iw', dest='en').text if isinstance(val, str) else val for val in col]

    def translate_df(dataframe):
        df_copy = dataframe.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == object:
                try:
                    df_copy[col] = translate_column(df_copy[col])
                except:
                    pass
        return df_copy

    with st.spinner("Translating and analyzing your data..."):
        df_translated = translate_df(df)

    # שלב 4: ניתוח בסיסי - דוגמה
    st.header("Basic Pattern Detection")

    if "Intensity" in df_translated.columns and "Type" in df_translated.columns:
        avg_by_type = df_translated.groupby("Type")["Intensity"].mean().sort_values()
        st.subheader("Average Intensity per Type")
        st.bar_chart(avg_by_type)
    else:
        st.warning("Column 'Intensity' not found in data.")
