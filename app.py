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
    "קפה": {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0},
    "סלמון": {"proteins": 25, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 8, "fats": 1, "carbohydrates": 83, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3, "fats": 1.5, "carbohydrates": 2, "dietaryFiber": 0.5},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "פלפל": {"proteins": 1, "fats": 0.2, "carbohydrates": 6, "dietaryFiber": 2},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
}

def extract_food_nutrition(food_name):
    total = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    for key in nutrition_db:
        if key in food_name:
            for k in total:
                total[k] += nutrition_db[key][k]
    return total

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

translated_data_global = []
translated_file_path = None

def translate_json(file_obj):
    global translated_data_global, translated_file_path
    try:
        content = file_obj.read().decode('utf-8')
        json_data = json.loads(content)
        translated_data = translate_value(json_data)

        for entry in translated_data:
            if isinstance(entry, dict) and "foodName" in entry:
                food_name = entry.get("foodName", "")
                values = extract_food_nutrition(food_name)
                entry["nutritionalValues"] = values

        translated_data_global = translated_data

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        translated_file_path = output_path
        return f"✅ File translated successfully. [Click here to download]({output_path})"
    except Exception as e:
        print("Error:", e)
        return "❌ Error during translation."

def generate_insights(month, mood_field, nutrition_field):
    if not translated_data_global:
        return "Please upload a file first."

    df = pd.DataFrame(translated_data_global)
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df[df["date"].dt.month == int(month)]

    df = df.dropna(subset=["date", mood_field, nutrition_field])
    df["hour"] = df["date"].dt.hour
    df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 10, 15, 24], labels=["morning", "noon", "evening"])

    group = df.groupby("time_of_day")[[mood_field, nutrition_field]].mean()

    insights = ""
    for time in group.index:
        mood_avg = round(group.loc[time][mood_field], 2)
        nut_avg = round(group.loc[time][nutrition_field], 2)
        insights += f"- During {time}: Mood avg = {mood_avg}, Nutrition avg ({nutrition_field}) = {nut_avg}\n"

    return insights if insights else "No insights found for selected data."

with gr.Blocks() as demo:
    gr.Markdown("### 🈯 JSON Hebrew to English Translator + Nutrition Enhancer")

    with gr.Row():
        file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
    download_link = gr.Markdown("⬇️ Download will appear here after upload.")

    file_input.change(fn=translate_json, inputs=file_input, outputs=download_link)

    gr.Markdown("---")
    gr.Markdown("### 📅 Select Month and Data Types for Insights")

    with gr.Row():
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], label="Select Month")
        mood_dropdown = gr.Dropdown(choices=["Parkinson's State", "My Mood", "Physical State"], label="Select Mood Field")
        nutrition_dropdown = gr.Dropdown(choices=["proteins", "fats", "carbohydrates", "dietaryFiber"], label="Select Nutrition Field")

    insights_output = gr.Textbox(label="📌 Insights", lines=6)
    analyze_btn = gr.Button("Generate Insights")

    analyze_btn.click(fn=generate_insights, inputs=[month_selector, mood_dropdown, nutrition_dropdown], outputs=insights_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
