import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os

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
    # המשך השלמה לפי הצורך
}

# מילון ערכים תזונתיים לפי רכיבים (לא כולל כמויות. רק ערכים ל-100 גרם או מנה שלמה)
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
    # המשך השלמה לפי הצורך
}

def extract_food_nutrition(food_name):
    # מפצל לפי מילים ומחפש רכיבים תזונתיים מוכרים
    total = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    for key in nutrition_db:
        if key in food_name:
            for k in total:
                total[k] += nutrition_db[key][k]
    return total

def translate_value(value, key=None):
    if key == "notes":
        return value  # skip notes field

    if isinstance(value, str):
        if value in translation_cache:
            return translation_cache[value]

        hebrew_chars = any('\u0590' <= c <= '\u05FF' for c in value)
        if hebrew_chars:
            try:
                result = translator.translate(value, "English")
                translation_cache[value] = result.result
                return result.result
            except Exception as e:
                print(f"Translation error for '{value}': {e}")
                return value
        return value
    elif isinstance(value, dict):
        return {k: translate_value(v, k) for k, v in value.items()}
    elif isinstance(value, list):
        return [translate_value(item) for item in value]
    else:
        return value

def translate_json(file_obj):
    if file_obj is None:
        return None

    try:
        content = file_obj.read().decode('utf-8')
        json_data = json.loads(content)

        # תרגום
        translated_data = translate_value(json_data)

        # עידכון ערכים תזונתיים אם מדובר בפריט תזונה
        for entry in translated_data:
            if isinstance(entry, dict) and "foodName" in entry:
                food_name = entry.get("foodName", "")
                values = extract_food_nutrition(food_name)
                entry["nutritionalValues"] = values

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        return output_path

    except Exception as e:
        print(f"Error translating JSON: {e}")
        return None

with gr.Blocks() as demo:
    gr.Markdown("### 🈯 JSON Hebrew to English Translator + Nutrition Enhancer")
    with gr.Row():
        file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
        output_file = gr.File(label="⬇️ Download Translated File")

    file_input.change(fn=translate_json, inputs=file_input, outputs=output_file)

    gr.Markdown("### 📊 Select what to include in analysis")
    with gr.Row():
        feelings_selector = gr.CheckboxGroup(
            choices=["Parkinson's State", "My Mood", "Physical State"],
            label="🧠 Feelings to Analyze",
            value=[]
        )
        types_selector = gr.CheckboxGroup(
            choices=["medicines", "nutritions", "activities", "symptoms"],
            label="📁 Data Types",
            value=[]
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
