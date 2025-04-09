
import streamlit as st
import pandas as pd
import json
import zipfile
import os

# מילון תרגום ידני ומהיר
translation_dict = {'מרגיש מצוין': 'Feeling great', 'איטיות': 'Slowness', 'איטי': 'Slow', 'לא מצליח להתאזן ולהתאמן': "Can't balance or exercise", 'בוקר טוב': 'Good morning', 'תחושה כללית פחות טובה': 'General feeling not so good', 'טוב': 'Good', 'קשה מאד': 'Very difficult', 'פיתה טחינה מלפפון עגבנייה ושניצל קטן': 'Pita with tahini, cucumber, tomato and small schnitzel', 'קערת קורנפלקס עם חלב סויה וצימוקים': 'Cereal bowl with soy milk and raisins', 'פלפל ומלפפון': 'Pepper and cucumber', 'שקדים טבעיים': 'Raw almonds', 'חצי פיתה עם חמאת בוטנים': 'Half pita with peanut butter', 'פלפל עם קוטג': 'Pepper with cottage cheese', 'מרק ירקות עם פתיתים': 'Vegetable soup with ptitim', 'עוגת תפוחים': 'Apple cake', 'קפה': 'Coffee', 'טנש': 'Tennis', 'עבודת גינה': 'Gardening', 'נסיעה לבית שאן': 'Trip to Beit Shean', 'אזילקט': 'Azilect', 'דופיקר 250': 'Dopicar 250'}

def translate_text(text):
    return translation_dict.get(text, text)

def translate_recursive(obj):
    if isinstance(obj, dict):
        return {translate_text(k): translate_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_recursive(i) for i in obj]
    elif isinstance(obj, str):
        return translate_text(obj)
    else:
        return obj

def load_json_from_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.json'):
                    with zip_ref.open(file_name) as json_file:
                        return json.load(json_file)
    elif uploaded_file.name.endswith('.json'):
        return json.load(uploaded_file)
    return None


def extract_all_types(data):
    types = set()
    for section in ["feelings", "symptoms", "activities"]:
        for item in data.get(section, []):
            if isinstance(item, dict) and "type" in item:
                types.add(item["type"])
    return sorted(types)

    types = set()
    for item in data.get("feelings", []):
        if "type" in item:
            types.add(item["type"])
    return sorted(types)


def filter_by_selected_types(data, selected_types):
    data = data.copy()
    for section in ["feelings", "symptoms", "activities"]:
        if section in data:
            data[section] = [item for item in data[section] if item.get("type") in selected_types]
    return data

    data = data.copy()
    if "feelings" in data:
        data["feelings"] = [f for f in data["feelings"] if f.get("type") in selected_types]
    return data

st.set_page_config(page_title="Fast Translator", layout="wide")
st.title("Parkinson's Analyzer (Fast Version)")

uploaded_file = st.file_uploader("Upload a JSON or ZIP file", type=["json", "zip"])
if uploaded_file:
    data = load_json_from_uploaded_file(uploaded_file)
    if data:
        types = extract_all_types(data)
        selected_types = st.multiselect("Choose types to analyze:", types, default=types)
        filtered_data = filter_by_selected_types(data, selected_types)
        translated_data = translate_recursive(filtered_data)

        # הורדה
        translated_path = "translated_data_fast.json"
        with open(translated_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        with open(translated_path, "rb") as f:
            st.download_button("📥 Download Translated JSON", f, file_name=translated_path)

        # תצוגה
        st.subheader("Filtered & Translated Feelings")
        df = pd.json_normalize(translated_data.get("feelings", []))
        st.dataframe(df)
    else:
        st.error("Failed to load JSON data.")
