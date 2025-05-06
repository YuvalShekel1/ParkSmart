import gradio as gr
import pandas as pd
import numpy as np
import json
import os
from collections import Counter
import tempfile

# נניח שפה כבר יש לך את:
# translated_data_global
translated_data_global = {}

# ---- פונקציות עזר ----

def prepare_activity_and_mood_data(data, mood_field):
    if not data or "activities" not in data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    activity_list = []
    for item in data.get("activities", []):
        if "date" in item:
            activity_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    activity_df = pd.DataFrame(activity_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return activity_df, mood_df

def prepare_medication_and_mood_data(data, mood_field):
    if not data or "medications" not in data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    medication_list = []
    for item in data.get("medications", []):
        if "date" in item:
            medication_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    medication_df = pd.DataFrame(medication_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return medication_df, mood_df

def prepare_symptom_and_mood_data(data, mood_field):
    if not data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    symptom_list = []
    for item in data.get("symptoms", []):
        if "date" in item:
            symptom_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    symptom_df = pd.DataFrame(symptom_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return symptom_df, mood_df

# ---- פונקציות ניתוח ----
# (כאן תכניס את הפונקציות המלאות שהבאת לי: generate_activity_insights, generate_medication_insights, generate_symptom_insights)

# (לצורך הדוגמה נכניס כאן הפניה פשוטה, אבל תשים בפועל את כל הקוד של הפונקציות שהבאת לי)

def generate_activity_insights(activity_df, mood_df):
    # כאן יהיה הקוד המלא של פונקציית ניתוח הפעילות שלך
    return "🏃 (כאן יוצג דוח פעילות אמיתי לפי מה שהבאת)"

def generate_medication_insights(medication_df, mood_df):
    # כאן יהיה הקוד המלא של פונקציית ניתוח התרופות שלך
    return "💊 (כאן יוצג דוח תרופות אמיתי לפי מה שהבאת)"

def generate_symptom_insights(symptom_df, mood_df):
    # כאן יהיה הקוד המלא של פונקציית ניתוח הסימפטומים שלך
    return "🩺 (כאן יוצג דוח סימפטומים אמיתי לפי מה שהבאת)"

# ---- פונקציות חיבור ל־Gradio ----

def activity_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    activity_df, mood_df = prepare_activity_and_mood_data(translated_data_global, mood_field)
    return generate_activity_insights(activity_df, mood_df)

def medication_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    medication_df, mood_df = prepare_medication_and_mood_data(translated_data_global, mood_field)
    return generate_medication_insights(medication_df, mood_df)

def symptom_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    symptom_df, mood_df = prepare_symptom_and_mood_data(translated_data_global, mood_field)
    return generate_symptom_insights(symptom_df, mood_df)

def upload_json(file):
    global translated_data_global
    if file is None:
        return "No file uploaded."
    
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        translated_data_global = data
        return "✅ File uploaded successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---- בניית הממשק ----

with gr.Blocks(title="Parkinson's Insights") as app:
    gr.Markdown("# 📈 Parkinson's Insights and Pattern Detection")
    file_input = gr.File(label="Upload JSON File")
    upload_button = gr.Button("Upload and Process File")
    upload_output = gr.Textbox(label="Upload Status")
    
    mood_selector = gr.Dropdown(
        ["Parkinson's State", "Physical State", "My Mood"],
        label="Select Mood Field",
        value="My Mood"
    )

    with gr.Tabs():
        with gr.TabItem("🏃 Activity Analysis"):
            activity_button = gr.Button("Analyze Activity Patterns")
            activity_output = gr.Markdown(label="Activity Insights")
        
        with gr.TabItem("💊 Medication Analysis"):
            medication_button = gr.Button("Analyze Medication Patterns")
            medication_output = gr.Markdown(label="Medication Insights")
        
        with gr.TabItem("🩺 Symptom Analysis"):
            symptom_button = gr.Button("Analyze Symptom Patterns")
            symptom_output = gr.Markdown(label="Symptom Insights")

    upload_button.click(fn=upload_json, inputs=[file_input], outputs=[upload_output])
    activity_button.click(fn=activity_analysis_summary, inputs=[mood_selector], outputs=[activity_output])
    medication_button.click(fn=medication_analysis_summary, inputs=[mood_selector], outputs=[medication_output])
    symptom_button.click(fn=symptom_analysis_summary, inputs=[mood_selector], outputs=[symptom_output])

# ---- הרצת האפליקציה ----

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.launch(server_name='0.0.0.0', server_port=port)
