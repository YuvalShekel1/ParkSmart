import streamlit as st
import pandas as pd
import json
import os
import zipfile
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Parkinson's Data Analyzer", layout="wide")
st.title("Parkinson's Data Analyzer")

translator = GoogleTranslator(source='auto', target='en')

def translate_text(text):
    if isinstance(text, str) and any('\u0590' <= c <= '\u05EA' for c in text):
        try:
            return translator.translate(text)
        except:
            return text
    return text

def translate_section(data_section):
    for item in data_section:
        for k, v in item.items():
            item[k] = translate_text(v)
    return data_section

def get_all_types(data):
    types = set()
    for section in ["feelings", "symptoms", "activities"]:
        for item in data.get(section, []):
            if isinstance(item, dict) and "type" in item:
                types.add(item["type"])
    return sorted(types)

def filter_data_by_types(data, selected_types):
    filtered = {}
    for section in ["feelings", "symptoms", "activities"]:
        if section in data:
            filtered[section] = [item for item in data[section] if item.get("type") in selected_types]
    return filtered

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

uploaded_file = st.file_uploader("Upload your JSON or ZIP file", type=["json", "zip"])
if uploaded_file:
    raw_data = load_json_from_uploaded_file(uploaded_file)
    if raw_data:
        all_types = get_all_types(raw_data)
        selected_types = st.multiselect("Select types to analyze:", all_types)

        if selected_types:
            filtered_data = filter_data_by_types(raw_data, selected_types)

            for section in ["feelings", "symptoms", "activities"]:
                if section in filtered_data:
                    filtered_data[section] = translate_section(filtered_data[section])

            translated_data = raw_data.copy()
            for section in filtered_data:
                translated_data[section] = filtered_data[section]

            translated_json = json.dumps(translated_data, ensure_ascii=False, indent=2)
            st.download_button("Download translated JSON", translated_json, file_name="translated_data.json", mime="application/json")

            st.success("Translation completed and file ready for download.")