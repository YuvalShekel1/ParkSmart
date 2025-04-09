import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from deep_translator import GoogleTranslator
import pandas as pd

# פונקציה לתרגום הקובץ כולו מעברית לאנגלית
def translate_json(file):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading file: {str(e)}", None

    # פונקציה שמתרגמת כל ערך
    def translate_value(val):
        if isinstance(val, str) and any("\u0590" <= ch <= "\u05EA" for ch in val):  # אם זה עברית
            return GoogleTranslator(source='he', target='en').translate(val)  # תרגום באמצעות deep_translator
        return val

    def recursive_translate(obj):
        if isinstance(obj, dict):
            return {k: recursive_translate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_translate(item) for item in obj]
        else:
            return translate_value(obj)

    # תרגום כל הנתונים
    translated = recursive_translate(data)

    # שמירת הנתונים המתורגמים כקובץ חדש
    translated_file_path = "/mnt/data/translated_data.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)

    return translated_file_path

# פונקציה ליצירת גרף
def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(0, 24, 1)  # 0-23 hours
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    for entry in data:
        hour = entry.get('hour', 0)
        value = entry.get('value', 3)

        # הצגת סימן עבור פעילות אם יש
        if selected_activity == 'nutritions' and hour == entry.get("hour"):
            ax.text(hour, value, symbol, fontsize=12, color="blue", ha='center')

        # הצגת נקודות עבור כל פרמטר שנבחר
        if "My Mood" in selected_types:
            ax.plot(hour, value, 'ro')
        if "Parkinson's State" in selected_types:
            ax.plot(hour, value, 'bo')
        if "Physical State" in selected_types:
            ax.plot(hour, value, 'go')

    plt.tight_layout()
    return fig

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    
    # בחירת סוגי "feelings" להצגה
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
    )

    # בחירת פעילות להצגה
    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
    )

    output_graph = gr.Plot(label="Graph of Mood and Activities")
    status_message = gr.HTML(label="Status", value="")
    download_button = gr.File(label="Download Translated File", file=None)

    # פונקציה להעלאת הקובץ ויצירת גרף
    def handle_upload(file, types, activity):
        status_message.update(value="Uploading and processing data... Please wait.")
        
        translated_file_path = translate_json(file)  # תרגום הקובץ
        if isinstance(translated_file_path, str) and translated_file_path.startswith("Error"):  # אם הייתה שגיאה
            status_message.update(value=translated_file_path)
            return None, status_message, None
        
        status_message.update(value="File uploaded and translated successfully!")
        
        # הצגת הגרף לאחר התרגום
        return plot_graph([], [types], activity), status_message, translated_file_path

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph, status_message, download_button])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
