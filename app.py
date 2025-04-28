import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os
import pandas as pd

translator = Translator()

# Translation cache
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

# Nutrition database
nutrition_db = {
    "פיתה": {"proteins": 6, "fats": 1.5, "carbohydrates": 33, "dietaryFiber": 1.5},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "קפה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 0.4, "dietaryFiber": 0},
    "סלמון": {"proteins": 25, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 7, "fats": 1, "carbohydrates": 84, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3.3, "fats": 2, "carbohydrates": 4, "dietaryFiber": 0.5},
    "חלב שקדים": {"proteins": 1.1, "fats": 2.5, "carbohydrates": 3, "dietaryFiber": 0.7},
    "צימוקים": {"proteins": 0.5, "fats": 0.2, "carbohydrates": 17, "dietaryFiber": 0.8},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "פלפל": {"proteins": 1, "fats": 0.3, "carbohydrates": 6, "dietaryFiber": 2.1},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
    "טחינה": {"proteins": 17, "fats": 57, "carbohydrates": 10, "dietaryFiber": 10},
}

translated_data_global = []
original_full_json = {}

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

def extract_food_nutrition(food_name):
    if food_name in nutrition_db:
        return nutrition_db[food_name]
    return {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}

def upload_and_process(file_obj):
    global translated_data_global, original_full_json

    try:
        file_path = file_obj.name
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_full_json = json.loads(content)

        # נניח שבקובץ יש שדה 'nutritions' או 'activities' או 'medications' וכו'
        keys_to_update = ["nutritions", "activities", "medications"]
        
        for key in keys_to_update:
            if key in original_full_json:
                section = original_full_json[key]
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict) and "foodName" in item:
                            food_name = item["foodName"]
                            item["nutritionalValues"] = extract_food_nutrition(food_name)
                
                # Translate the section
                original_full_json[key] = translate_value(section)

        translated_data_global = original_full_json

        # Save translated full JSON
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)

        return output_path, "✅ File processed successfully!"

    except Exception as e:
        return None, f"❌ Error processing: {str(e)}"

def generate_insights(year, month, mood_field, nutrition_field):
    if not translated_data_global:
        return "Please upload a file first."

    try:
        df = pd.DataFrame(translated_data_global.get("nutritions", []))

        if df.empty:
            return "No nutrition data found."

        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df = df[(df["date"].dt.year == int(year)) & (df["date"].dt.month == int(month))]

        if df.empty:
            return "No data for selected month and year."

        if nutrition_field not in df.columns:
            return f"Field {nutrition_field} not found."

        insights = f"Data analysis for {month}/{year}\n"
        insights += f"Average {mood_field}: {df[mood_field].mean()}\n"
        insights += f"Average {nutrition_field}: {df[nutrition_field].mean()}\n"

        return insights

    except Exception as e:
        return f"Error generating insights: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 🈯 JSON Translator + Full Data Nutrition Update")

    with gr.Row():
        file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
        output_file = gr.File(label="⬇️ Download updated JSON")

    file_input.change(fn=upload_and_process, inputs=file_input, outputs=[output_file, gr.Textbox(interactive=False)])

    gr.Markdown("---")
    gr.Markdown("## 📅 Analyze Mood and Nutrition")

    with gr.Row():
        year_selector = gr.Dropdown(choices=["2024", "2025"], label="Select Year")
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], label="Select Month")
        mood_dropdown = gr.Dropdown(choices=["Parkinson's State", "My Mood", "Physical State"], label="Select Mood Field")
        nutrition_dropdown = gr.Dropdown(choices=["proteins", "fats", "carbohydrates", "dietaryFiber"], label="Select Nutrition Field")

    insights_output = gr.Textbox(label="📌 Insights", lines=8)
    analyze_btn = gr.Button("🔍 Generate Insights")

    analyze_btn.click(
        fn=generate_insights,
        inputs=[year_selector, month_selector, mood_dropdown, nutrition_dropdown],
        outputs=insights_output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
