
import streamlit as st
import pandas as pd
import json
import os
import zipfile
from deep_translator import GoogleTranslator

st.title("Parkinson's Data Analyzer")

uploaded_file = st.file_uploader("Upload your JSON or ZIP file", type=["json", "zip"])

translator = GoogleTranslator(source='auto', target='en')

def translate_text(text):
    if isinstance(text, str) and any('֐' <= c <= 'ת' for c in text):
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

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_ext == ".zip":
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            filenames = zip_ref.namelist()
            json_filename = [f for f in filenames if f.endswith('.json')][0]
            with zip_ref.open(json_filename) as json_file:
                raw_data = json.load(json_file)
    elif file_ext == ".json":
        raw_data = json.load(uploaded_file)
    else:
        st.error("Unsupported file type")
        st.stop()

    all_types = get_all_types(raw_data)
    selected_types = st.multiselect("Select types to analyze:", all_types)

    if selected_types:
        filtered_data = filter_data_by_types(raw_data, selected_types)

        for section in ["feelings", "symptoms", "activities"]:
            if section in filtered_data:
                filtered_data[section] = translate_section(filtered_data[section])

        # Insert back into full structure for completeness
        translated_data = raw_data.copy()
        for section in filtered_data:
            translated_data[section] = filtered_data[section]

        translated_json = json.dumps(translated_data, ensure_ascii=False, indent=2)
        st.download_button("Download translated JSON", translated_json, file_name="translated_data.json", mime="application/json")

        st.success("Translation completed and file ready for download.")
