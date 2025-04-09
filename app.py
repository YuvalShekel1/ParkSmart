import gradio as gr
import json
import os
from deep_translator import GoogleTranslator
import requests
from github import Github

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
    github_token = "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"  # הכנס את הטוקן שלך
    repo_name = "your_github_username/your_repository_name"  # הכנס את שם הרפוזיטורי
    file_path_in_repo = ".github/workflows/translated_data.json"  # נתיב הקובץ ב-GitHub
    
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

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # העלאת קובץ JSON עם כפתור קטן
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"])
    
    # פונקציה להעלאת הקובץ
    def handle_upload(file):
        file_url = translate_json(file)  # תרגום הקובץ
        return gr.HTML(f'<a href="{file_url}" target="_blank">Download Translated File</a>')  # הצגת קישור להורדה

    translate_btn = gr.Button("Generate Translated File")
    translate_btn.click(fn=handle_upload, inputs=[file_input], outputs=[gr.HTML()])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
