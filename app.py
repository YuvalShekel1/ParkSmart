import gradio as gr
import json
import tempfile
import os
import re
from datetime import datetime
from translatepy import Translator
from openai import OpenAI
import pandas as pd

translator = Translator()
openai_client = OpenAI()

translation_cache = {
    "פיתה": "Pita",
    "חמאת בוטנים": "Peanut butter",
    # הוספה חלקית – המשיכי להרחיב
}

# מילון ערכים תזונתיים בסיסיים ליחידה אחת של מאכל
nutrition_dict = {
    "פיתה": {"proteins": 6, "fats": 1, "carbohydrates": 30, "dietaryFiber": 2},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2}
}

def extract_ingredients(text):
    items = re.split(r'[.,\s]+', text)
    return [item for item in items if item]

def get_nutrition_from_text(food_text):
    ingredients = extract_ingredients(food_text)
    total = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    for item in ingredients:
        for food in nutrition_dict:
            if food in item:
                portion = 1.0
                if "חצי" in item:
                    portion = 0.5
                if "רבע" in item:
                    portion = 0.25
                for key in total:
                    total[key] += nutrition_dict[food][key] * portion
                break
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
            except Exception:
                return value
        return value
    elif isinstance(value, dict):
        return {k: translate_value(v, k) for k, v in value.items()}
    elif isinstance(value, list):
        return [translate_value(item) for item in value]
    return value

def enrich_nutrition(data):
    for entry in data:
        if entry.get("type") == "nutrition" and "foodName" in entry:
            values = get_nutrition_from_text(entry["foodName"])
            entry["nutritionalValues"] = values
    return data

def translate_json(file_obj):
    if file_obj is None:
        return None
    try:
        content = file_obj.read().decode('utf-8')
        json_content = json.loads(content)
        translated_json = translate_value(json_content)
        enriched_json = enrich_nutrition(translated_json)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_json, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        print(f"Error translating JSON: {e}")
        return None

def get_year_month_options(data):
    dates = []
    for entry in data:
        if "dateTaken" in entry:
            try:
                dt = datetime.fromisoformat(entry["dateTaken"].replace("Z", ""))
                dates.append((dt.year, dt.month))
            except:
                continue
    return sorted({f"{y}-{m:02}" for y, m in dates})

def extract_months(file):
    if file is None:
        return []
    content = file.read().decode("utf-8")
    data = json.loads(content)
    return gr.update(choices=get_year_month_options(data))

def analyze_month(file, month_year):
    content = file.read().decode("utf-8")
    data = json.loads(content)
    df = pd.DataFrame(data)
    df["dateTaken"] = pd.to_datetime(df["dateTaken"], errors="coerce")
    df = df[df["dateTaken"].dt.strftime("%Y-%m") == month_year]

    mood = df[df["type"] == "My Mood"]
    nutrition = df[df["type"] == "nutrition"]

    if mood.empty or nutrition.empty:
        return "⚠️ Not enough data for insights."

    mood["date"] = mood["dateTaken"].dt.date
    nutrition["hour"] = nutrition["dateTaken"].dt.hour
    nutrition["date"] = nutrition["dateTaken"].dt.date

    morning = nutrition[nutrition["hour"].between(5, 10)]
    days_with_protein = morning[morning["nutritionalValues"].apply(lambda x: x["proteins"] > 0)]["date"].unique()
    with_protein = mood[mood["date"].isin(days_with_protein)]["value"].astype(float)
    without_protein = mood[~mood["date"].isin(days_with_protein)]["value"].astype(float)

    avg_with = with_protein.mean()
    avg_without = without_protein.mean()

    if pd.isna(avg_with) or pd.isna(avg_without):
        return "⚠️ Not enough mood data."

    diff = round(avg_with - avg_without, 2)
    if diff > 0:
        return f"✔ Eating proteins in the morning improved mood by {diff} points on average."
    elif diff < 0:
        return f"✖ Eating proteins in the morning lowered mood by {abs(diff)} points."
    else:
        return "➖ No significant mood change related to morning protein intake."

with gr.Blocks(css=".gr-box {padding: 20px;}") as demo:
    gr.Markdown("### 🈯 Upload JSON for Automatic Translation + Nutrition Analysis")
    with gr.Row():
        file_input = gr.File(label="📤 Upload JSON", file_types=[".json"])
        output_file = gr.File(label="📥 Download Translated File")

    file_input.change(fn=translate_json, inputs=file_input, outputs=output_file)

    gr.Markdown("### 🎯 Select Analysis Parameters")
    feelings_selector = gr.CheckboxGroup(["Parkinson's State", "My Mood", "Physical State"], label="Feelings")
    types_selector = gr.CheckboxGroup(["medicines", "nutritions", "activities", "symptoms"], label="Data Types")

    gr.Markdown("### 📊 Monthly Insights")
    month_dropdown = gr.Dropdown(label="Choose month-year", choices=[])
    insight_box = gr.Textbox(label="Insights", lines=4)

    file_input.change(fn=extract_months, inputs=file_input, outputs=month_dropdown)
    month_dropdown.change(fn=analyze_month, inputs=[file_input, month_dropdown], outputs=insight_box)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
