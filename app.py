import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# פונקציה לתרגום של ערכים בעברית לאנגלית
def translate_value(val):
    if isinstance(val, str) and any("\u0590" <= ch <= "\u05EA" for ch in val):  # אם זה טקסט בעברית
        return f"translated({val})"  # כאן צריך לשים את הלוגיקה של תרגום אמיתי לעברית לאנגלית
    return val

# פונקציה שמבצעת את התרגום לכל הקובץ
def translate_json(file):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading file: {str(e)}", None

    def recursive_translate(obj):
        if isinstance(obj, dict):
            return {k: recursive_translate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_translate(item) for item in obj]
        else:
            return translate_value(obj)

    translated_data = recursive_translate(data)
    return translated_data

# פונקציה להציג גרף
def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    fig, ax = plt.subplots(figsize=(10, 6))

    hours = np.arange(0, 24, 1)  # 0-23 hours
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    for entry in data:
        hour = entry.get('hour', 0)
        value = entry.get('value', 3)

        if selected_activity == 'nutritions' and hour == entry.get("hour"):
            ax.text(hour, value, symbol, fontsize=12, color="blue", ha='center')

        if "My Mood" in selected_types:
            ax.plot(hour, value, 'ro')
        if "Parkinson's State" in selected_types:
            ax.plot(hour, value, 'bo')
        if "Physical State" in selected_types:
            ax.plot(hour, value, 'go')

    plt.tight_layout()
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    gr.HTML("""
        <style>
            /* Custom style for the file upload button only */
            #file-upload-btn {
                font-size: 12px;  /* Small font size */
                padding: 5px 10px;  /* Smaller padding */
                height: 40px;  /* Smaller height */
                width: 150px;  /* Smaller width */
                border-radius: 5px; /* Rounded corners for square button */
                margin-bottom: 10px;  /* Optional: Space below the button */
            }
        </style>
    """)

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    
    # Radio buttons לבחירת רגשות לתצוגה
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
    )

    # Radio buttons לבחירת פעילות להציג
    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
    )

    # גרף של רגשות ופעילויות
    output_graph = gr.Plot(label="Graph of Mood and Activities")
    status_message = gr.HTML(label="Status", value="")

    def handle_upload(file, types, activity):
        # הצגת פידבק למשתמש
        status_message.update(value="Uploading and processing data... Please wait.")
        
        # תרגום הקובץ
        translated_data = translate_json(file)
        if isinstance(translated_data, str):  # אם יש שגיאה
            status_message.update(value=translated_data)
            return None  # לא ליצור גרף אם הייתה שגיאה
        
        # עדכון מצב אחרי שהקובץ הועלה בהצלחה
        status_message.update(value="File uploaded and translated successfully!")
        
        # יצירת הגרף
        return plot_graph(translated_data, [types], activity)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph, status_message])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
