import gradio as gr
import json
import os
from deep_translator import GoogleTranslator
import time

# פונקציה לתרגום הקובץ כולו מעברית לאנגלית
def translate_json(file, progress=gr.Progress()):
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
            try:
                # וידוא שהשפות נכונות — deep_translator מקבלת קודים של שפות
                return GoogleTranslator(source='he', target='en').translate(val)  # תרגום באמצעות deep_translator
            except Exception as e:
                return f"Error during translation: {str(e)}"
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

    # יצירת תיקיית היעד אם היא לא קיימת
    output_dir = "outputs/"  # תיקיית פלט בפרויקט, אותה יוכל ה־GitHub Actions לגשת אליה
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # שמירת הקובץ המתורגם
    translated_file_path = os.path.join(output_dir, "translated_data.json")
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)

    progress(1)  # סיום, 100%

    return translated_file_path

# פונקציה ליצירת גרף
def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    # יצירת גרף לדוגמה (לפי הצורך שלך)
    return None  # יש להחליף בקוד יצירת הגרף שלך

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"])
    
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
    progress_bar = gr.Progress()  # הסרה של פרמטר label

    # פונקציה להעלאת הקובץ ויצירת גרף
    def handle_upload(file, types, activity):
        status_message.value = "Uploading and processing data... Please wait."  # עדכון סטטוס
        
        # הצגת בר הטעינה במהלך התרגום
        translated_file_path = translate_json(file, progress_bar)  # תרגום הקובץ
        if isinstance(translated_file_path, str):  # אם התשובה היא הודעת שגיאה
            status_message.value = translated_file_path
            return None, status_message, None  # מחזירים None עבור הגרף והקובץ
        
        status_message.value = "File uploaded and translated successfully!"
        
        # הצגת הגרף לאחר התרגום
        return plot_graph([], [types], activity), status_message, gr.File.update(value=translated_file_path, visible=True)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph, status_message, file_input])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
