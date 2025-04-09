import gradio as gr
import json
import os
from deep_translator import GoogleTranslator
import requests
from github import Github
import matplotlib.pyplot as plt
import numpy as np

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
            try:
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

    # יצירת הקובץ המתורגם
    translated_file_path = "translated_data.json"
    with open(translated_file_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=4)

    # העלאת הקובץ ל-GitHub
    github_token = os.getenv("GITHUB_TOKEN")  # טוקן GitHub מתוך משתנה סביבה
    repo_name = "YuvalShekel1/ParkSmart"  # שם הרפוזיטורי שלך
    file_path_in_repo = "files/translated_data.json"  # נתיב הקובץ ב-GitHub
    
    # אתחול של GitHub API
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    
    # קריאת הקובץ על מנת להעלות אותו מחדש
    with open(translated_file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # העלאת הקובץ ל-GitHub
    repo.create_file(file_path_in_repo, "Upload translated file", content)

    # יצירת קישור להורדה
    file_url = f"https://github.com/{repo_name}/blob/main/{file_path_in_repo}"
    
    return file_url

# פונקציה ליצירת גרף ריק
def create_empty_graph():
    x = np.arange(0, 24, 1)  # שעות מ-0 עד 23 (12 בלילה עד 12 בלילה)
    y = np.zeros_like(x)  # נתונים ריקים לציר Y (ערכים שווים לאפס)

    plt.figure(figsize=(10,6))
    plt.plot(x, y, label="Empty Graph")
    plt.xlabel("Hours of the Day (12 AM to 12 AM)")
    plt.ylabel("Values (1-5)")
    plt.title("Empty Graph with X and Y Axis")
    plt.xticks(np.arange(0, 24, 1), labels=[f"{int(i)}:00" for i in np.arange(0, 24, 1)])
    plt.yticks(np.arange(1, 6, 1))
    plt.grid(True)
    plt.legend()
    
    # שמירת הגרף כקובץ תמונה
    graph_path = "/mnt/data/empty_graph.png"
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"])

    # הצגת גרף ריק
    empty_graph = gr.Image(label="Empty Graph", type="file")

    # פונקציה להעלאת הקובץ
    def handle_upload(file):
        file_url = translate_json(file)  # תרגום הקובץ
        graph_path = create_empty_graph()  # יצירת גרף ריק
        return gr.HTML(f'<a href="{file_url}" target="_blank">Download Translated File</a>'), graph_path  # הצגת קישור להורדה + גרף

    translate_btn = gr.Button("Generate Translated File and Graph")
    translate_btn.click(fn=handle_upload, inputs=[file_input], outputs=[gr.HTML(), empty_graph])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
