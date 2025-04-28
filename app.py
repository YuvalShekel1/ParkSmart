import gradio as gr
import json
import tempfile
from translatepy import Translator
import os
from datetime import datetime

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
    "Bowl of cornflakes with soy milk and raisins": {"proteins": 10.8, "fats": 3.2, "carbohydrates": 105, "dietaryFiber": 4.3},
    "Half pita with peanut butter": {"proteins": 11, "fats": 16.75, "carbohydrates": 22.5, "dietaryFiber": 2.75},
    "Quarter pita with peanut spread": {"proteins": 9.5, "fats": 16.38, "carbohydrates": 14.25, "dietaryFiber": 2.38},
    "Salmon with mashed potatoes and peas": {"proteins": 32, "fats": 14.5, "carbohydrates": 29, "dietaryFiber": 6.5},
    "Pita with tahini, cucumber, tomato and schnitzel": {"proteins": 33.6, "fats": 27.8, "carbohydrates": 49.4, "dietaryFiber": 5.2},
}

def translate_text(text):
    if not isinstance(text, str):
        return text
    if text in translation_cache:
        return translation_cache[text]
    if any('\u0590' <= c <= '\u05FF' for c in text):
        try:
            translated = translator.translate(text, "English")
            return translated.result
        except:
            return text
    return text

def update_nutrition(entry):
    """Update nutrition if the entry has a 'foodName'."""
    if isinstance(entry, dict) and "foodName" in entry:
        food_name = entry["foodName"]
        values = nutrition_db.get(food_name, {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0})
        entry["nutritionalValues"] = values
    return entry

def recursive_translate_and_update(data):
    """Recursively translate and update the JSON data without changing its structure."""
    if isinstance(data, dict):
        return {k: recursive_translate_and_update(update_nutrition(v) if k == "nutritions" or k == "nutrition" else v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_translate_and_update(update_nutrition(item)) for item in data]
    else:
        return translate_text(data)

def upload_and_translate(file_obj):
    try:
        content = file_obj.read().decode('utf-8')
        json_data = json.loads(content)

        translated_data = recursive_translate_and_update(json_data)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        return output_path
    except Exception as e:
        print("Error:", e)
        return None

with gr.Blocks() as demo:
    gr.Markdown("# 🈯 Upload JSON - Translate + Update Nutrition")

    with gr.Row():
        file_input = gr.File(label="⬆️ Upload JSON File", file_types=[".json"])
        output_file = gr.File(label="⬇️ Download Updated JSON")

    file_input.change(fn=upload_and_translate, inputs=[file_input], outputs=[output_file])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
