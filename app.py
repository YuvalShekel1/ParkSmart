import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def translate_json(file, selected_types):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def translate_value(val):
        # תרגום עברית לאנגלית
        if isinstance(val, str) and any("\u0590" <= ch <= "\u05EA" for ch in val):  # Detect Hebrew
            # תרגום דוגמה
            return f"translated({val})"  # זו רק דוגמה לתרגום
        return val

    def recursive_translate(obj):
        if isinstance(obj, dict):
            return {k: recursive_translate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_translate(item) for item in obj]
        else:
            return translate_value(obj)

    # סינון המידע על פי סוגי הטייפ שנבחרו
    translated = [
        recursive_translate(item) for item in data if item.get("type") in selected_types
    ]

    return translated

def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    # יצירת גרף
    fig, ax = plt.subplots(figsize=(10, 6))

    # יצירת ציר X של שעות (12 בלילה עד 12 בצהריים)
    hours = np.arange(0, 24, 1)  # שעות 0-23
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    # יצירת ציר Y עם ערכים מ-1 עד 5 (FEELINGS)
    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    # כותרת לגרף
    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    # יצירת הגרף עבור כל פרמטר שנבחר
    for entry in data:
        # זיהוי שעה ונתון של המשתמש
        hour = entry.get('hour', 0)  # שעה (מ-0 ל-23)
        value = entry.get('value', 3)  # ערך של ההרגשה (1-5)

        # הצגת סימן כמו כוס קפה (או משהו אחר) עבור סוגי פעילות
        if selected_activity == 'nutritions' and hour == entry.get("hour"):
            ax.text(hour, value, symbol, fontsize=12, color="blue", ha='center')

        # יצירת הגרף עבור כל פרמטר שנבחר
        if "My Mood" in selected_types:
            ax.plot(hour, value, 'ro')  # הצגת נקודה אדומה עבור mood
        if "Parkinson's State" in selected_types:
            ax.plot(hour, value, 'bo')  # הצגת נקודה כחולה עבור Parkinson's State
        if "Physical State" in selected_types:
            ax.plot(hour, value, 'go')  # הצגת נקודה ירוקה עבור Physical State

    # הצגת גרף
    plt.tight_layout()
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    
    # תיבת צ'קבוקס לבחירת סוגי טייפ לתרגום (בחר אפשרות אחת)
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
    )

    # תיבת צ'קבוקס לבחירת פעילות להציג
    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
    )

    # שדה תצוגת גרף
    output_graph = gr.Plot(label="Graph of Mood and Activities")

    def handle_upload(file, types, activity):
        data = translate_json(file, types)
        return plot_graph(data, [types], activity)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph])

    # Set the app to listen on all IPs and the port provided by Render
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
