## מבחינת עיצוב סרגל כלים וכל זה זה מעולה! רק אין את כל עניין התובנות והתרגום והערכים תזונתיים והורדה תודה של הקובץ 

import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import tempfile
from collections import Counter

# גלובלים
translated_data_global = {}
processed_file_path = ""

# --- עזר: הכנת הדאטה פריים ---

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

# --- ניתוחים ודפוסים ---

def generate_activity_insights(activity_df, mood_df):
    insights = "🏃 Activity Insights:\n"

    if activity_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze activities and mood correlation.\n"

    combined_data = []
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        same_day_activities = activity_df[activity_df["date"].dt.date == mood_date.date()]
        for _, act_row in same_day_activities.iterrows():
            activity_item = act_row["item"]
            time_diff = 24
            if "startTime" in activity_item:
                try:
                    act_time = pd.to_datetime(activity_item["startTime"])
                    time_diff = abs((mood_date - act_time).total_seconds() / 3600)
                except:
                    pass
            if time_diff <= 6:
                activity_type = activity_item.get("activityType", "Unknown")
                duration = activity_item.get("duration", 0)
                intensity = activity_item.get("intensity", 0)
                combined_data.append({
                    "mood_value": mood_value,
                    "activity_type": activity_type,
                    "duration": duration,
                    "intensity": intensity,
                    "time_diff": time_diff
                })

    if not combined_data or len(combined_data) < 2:
        return insights + "Not enough close-timing data.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• Activity type impact on mood:\n"
    activity_types = analysis_df["activity_type"].unique()
    for activity in activity_types:
        act_mood = analysis_df[analysis_df["activity_type"] == activity]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = act_mood - overall_mood
        if not np.isnan(act_mood):
            count = len(analysis_df[analysis_df["activity_type"] == activity])
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.5:
                insights += f"  - {activity} ({count} times): Mood tends to be {abs(round(diff, 1))} points {direction} than average.\n"
            else:
                insights += f"  - {activity} ({count} times): Mood similar to average.\n"

    return insights

def generate_medication_insights(medication_df, mood_df):
    insights = "💊 Medication Insights:\n"

    if medication_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze medications.\n"

    combined_data = []
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        same_day_meds = medication_df[medication_df["date"].dt.date == mood_date.date()]
        for _, med_row in same_day_meds.iterrows():
            med_item = med_row["item"]
            time_diff = 24
            if "timeTaken" in med_item:
                try:
                    med_time = pd.to_datetime(med_item["timeTaken"])
                    time_diff = abs((mood_date - med_time).total_seconds() / 3600)
                except:
                    pass
            if time_diff <= 6:
                med_name = med_item.get("medicationName", "Unknown")
                dosage = med_item.get("dosage", 0)
                combined_data.append({
                    "mood_value": mood_value,
                    "medication": med_name,
                    "dosage": dosage,
                    "time_diff": time_diff
                })

    if not combined_data or len(combined_data) < 2:
        return insights + "Not enough close-timing data.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• Medication impact on mood:\n"
    medications = analysis_df["medication"].unique()
    for med in medications:
        med_mood = analysis_df[analysis_df["medication"] == med]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = med_mood - overall_mood
        if not np.isnan(med_mood):
            count = len(analysis_df[analysis_df["medication"] == med])
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.3:
                insights += f"  - {med} ({count} times): Mood tends to be {abs(round(diff, 1))} points {direction}.\n"
            else:
                insights += f"  - {med} ({count} times): Mood similar to average.\n"

    return insights

def generate_symptom_insights(symptom_df, mood_df):
    insights = "🩺 Symptom Insights:\n"

    if symptom_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze symptoms.\n"

    symptom_fields = set()
    for _, row in symptom_df.iterrows():
        item = row["item"]
        for key in item.keys():
            if key not in ["date", "notes", "id", "Parkinson's State", "My Mood", "Physical State"]:
                symptom_fields.add(key)
    symptom_fields = list(symptom_fields)

    if not symptom_fields:
        return insights + "No specific symptom fields detected.\n"

    insights += "• Symptom impact on mood:\n"
    date_to_mood = {row["date"].date(): row["value"] for _, row in mood_df.iterrows()}

    for symptom in symptom_fields:
        symptom_present_moods = []
        symptom_absent_moods = []
        for _, row in symptom_df.iterrows():
            date = row["date"].date()
            item = row["item"]
            if date in date_to_mood:
                mood_value = date_to_mood[date]
                if symptom in item and item[symptom]:
                    symptom_present_moods.append(mood_value)
                else:
                    symptom_absent_moods.append(mood_value)
        if symptom_present_moods and symptom_absent_moods:
            present_avg = np.mean(symptom_present_moods)
            absent_avg = np.mean(symptom_absent_moods)
            diff = present_avg - absent_avg
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.3:
                insights += f"  - {symptom}: Mood {direction} by {round(abs(diff),1)} points when present.\n"
            else:
                insights += f"  - {symptom}: No strong mood impact.\n"

    return insights

# --- חיבור לגרידיו ---

def upload_json(file_obj):
    global translated_data_global, processed_file_path
    if file_obj is None:
        return None, "❌ No file uploaded."
    try:
        content = file_obj.read()
        data = json.loads(content)
        translated_data_global = data
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(temp_path.name, "w", encoding="utf-8") as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)
        processed_file_path = temp_path.name
        return processed_file_path, "✅ File uploaded and processed."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

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

# --- יצירת האפליקציה ---

with gr.Blocks(title="Parkinson's Health Pattern Analysis") as app:
    gr.Markdown("# 📈 Parkinson's Health Pattern Analysis")

    with gr.Row():
        file_input = gr.File(label="Upload JSON File")
        upload_button = gr.Button("Upload and Process")
    
    output_text = gr.Textbox(label="Status", interactive=False)
    processed_file = gr.File(label="Download Processed File", interactive=False)

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

    upload_button.click(fn=upload_json, inputs=[file_input], outputs=[processed_file, output_text])
    activity_button.click(fn=activity_analysis_summary, inputs=[mood_selector], outputs=[activity_output])
    medication_button.click(fn=medication_analysis_summary, inputs=[mood_selector], outputs=[medication_output])
    symptom_button.click(fn=symptom_analysis_summary, inputs=[mood_selector], outputs=[symptom_output])

# --- הפעלת האפליקציה ---

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.launch(server_name='0.0.0.0', server_port=port)
