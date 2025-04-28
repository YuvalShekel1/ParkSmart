import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os
import pandas as pd

translator = Translator()

# מילון תרגום
translation_cache = {
    "איטי": "Slow",
    "לא מצליח להתאזן ולהתאמן": "Unable to balance and exercise",
    "בוקר טוב": "Good morning",
    "תחושה כללית פחות טובה": "General feeling is less good",
    "מרגיש מצוין": "Feeling excellent",
    "איטיות": "Slowness",
    "טוב": "Good",
    "התכווצויות בכפות הרגליים למשך 15 דקות": "Foot cramps for 15 minutes",
    "התכווצויות באצבעות רגל ימין": "Toe cramps in right foot",
    "אזילקט": "Azilect",
    "דופיקר": "Dopicar",
    "דופיקר 125": "Dopicar 125",
    "דופיקר 175": "Dopicar 175",
    "דופיקר 250": "Dopicar 250",
    "קפה": "Coffee",
    "חצי פיתה עם חמאת בוטנים": "Half pita with peanut butter",
    "פלפל ומלפפון": "Pepper and cucumber",
    "קערת קורנפלקס עם חלב סויה וצימוקים": "Bowl of cornflakes with soy milk and raisins",
    "סלמון עם פירה ואפונה": "Salmon with mashed potatoes and peas",
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": "Pita with tahini, cucumber, tomato and schnitzel",
}

# מילון ערכים תזונתיים
nutrition_db = {
    "פיתה": {"proteins": 6, "fats": 1.5, "carbohydrates": 33, "dietaryFiber": 1.5},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "קפה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 0.4, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 7, "fats": 1, "carbohydrates": 84, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3.3, "fats": 2, "carbohydrates": 4, "dietaryFiber": 0.5},
    "צימוקים": {"proteins": 0.5, "fats": 0.2, "carbohydrates": 17, "dietaryFiber": 0.8},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "פלפל": {"proteins": 1, "fats": 0.3, "carbohydrates": 6, "dietaryFiber": 2.1},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
    "טחינה": {"proteins": 17, "fats": 57, "carbohydrates": 10, "dietaryFiber": 10},
    "עגבנייה": {"proteins": 0.9, "fats": 0.2, "carbohydrates": 3.9, "dietaryFiber": 1.2},
    "פירה": {"proteins": 2, "fats": 0.1, "carbohydrates": 15, "dietaryFiber": 1.5},
    "אפונה": {"proteins": 5, "fats": 0.4, "carbohydrates": 14, "dietaryFiber": 5},
    "Bowl of cornflakes with soy milk and raisins": {"proteins": 10.8, "fats": 3.2, "carbohydrates": 105, "dietaryFiber": 4.3},
    "Half pita with peanut butter": {"proteins": 11, "fats": 16.75, "carbohydrates": 22.5, "dietaryFiber": 2.75},
    "Salmon with mashed potatoes and peas": {"proteins": 32, "fats": 14.5, "carbohydrates": 29, "dietaryFiber": 6.5},
    "Pita with tahini, cucumber, tomato and schnitzel": {"proteins": 33.6, "fats": 27.8, "carbohydrates": 49.4, "dietaryFiber": 5.2},
}

translated_data_global = {}

def extract_food_nutrition(food_name):
    return nutrition_db.get(food_name, {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0})

def translate_value(value, key=None):
    if key == "notes":
        return value
    if isinstance(value, str):
        if value in translation_cache:
            return translation_cache[value]
        hebrew_chars = any('\u0590' <= c <= '\u05FF' for c in value)
        if hebrew_chars:
            try:
                result = translator.translate(value, "English")
                translation_cache[value] = result.result
                return result.result
            except:
                return value
        return value
    elif isinstance(value, dict):
        return {k: translate_value(v, k) for k, v in value.items()}
    elif isinstance(value, list):
        return [translate_value(item) for item in value]
    else:
        return value

def upload_and_process(file_obj):
    global translated_data_global
    try:
        file_path = file_obj.name
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)

        # נתרגם הכל
        translated = translate_value(data)

        # נטפל בערכים של nutritions
        if isinstance(translated, dict) and "nutritions" in translated:
            for entry in translated["nutritions"]:
                if "foodName" in entry:
                    entry["nutritionalValues"] = extract_food_nutrition(entry["foodName"])

        translated_data_global = translated

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, ensure_ascii=False)
        return output_path, "✅ Translation and nutrition update complete."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def generate_insights(year, month, mood_field, nutrition_field):
    if not translated_data_global:
        return "Please upload a file first."
    try:
        df = pd.DataFrame(translated_data_global.get("feelings", []))
        df["date"] = pd.to_datetime(df.get("dateTaken", df.get("createdAt", df.get("date"))), errors='coerce')
        df = df[(df["date"].dt.year == int(year)) & (df["date"].dt.month == int(month))]
        
        if "nutritionalValues" in df.columns:
            df = pd.concat([df.drop(["nutritionalValues"], axis=1), df["nutritionalValues"].apply(pd.Series)], axis=1)

        if mood_field not in df.columns or nutrition_field not in df.columns:
            return "Selected fields not found in data."

        mood_avg = df[mood_field].mean()
        nutrition_avg = df[nutrition_field].mean()
        return f"Average {mood_field}: {round(mood_avg, 2)}\nAverage {nutrition_field}: {round(nutrition_avg, 2)}"
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🈯 JSON Translator + Nutrition Updater")

    with gr.Row():
        file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
        output_file = gr.File(label="⬇️ Download Translated JSON")
    status_text = gr.Textbox(label="Status", interactive=False)

    file_input.change(fn=upload_and_process, inputs=[file_input], outputs=[output_file, status_text])

    gr.Markdown("---")
    gr.Markdown("## 📅 Analyze Mood vs Nutrition")

    with gr.Row():
        year_selector = gr.Dropdown(choices=["2024", "2025"], label="Select Year")
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], label="Select Month")
    
    with gr.Row():
        mood_dropdown = gr.Dropdown(choices=["Parkinson's State", "My Mood", "Physical State"], label="Select Mood Field")
        nutrition_dropdown = gr.Dropdown(choices=["proteins", "fats", "carbohydrates", "dietaryFiber"], label="Select Nutrition Field")

    insights_output = gr.Textbox(label="📌 Insights", lines=8)
    analyze_btn = gr.Button("🔍 Generate Insights")

    analyze_btn.click(fn=generate_insights, inputs=[year_selector, month_selector, mood_dropdown, nutrition_dropdown], outputs=insights_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
