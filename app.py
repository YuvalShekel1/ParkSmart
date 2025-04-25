import gradio as gr
import json
import tempfile
import pandas as pd
from datetime import datetime

from translatepy import Translator
translator = Translator()

# מילון תרגום
translation_cache = {
    "קפה": "Coffee",
    "חצי פיתה עם חמאת בוטנים": "Half pita with peanut butter",
    "קערת קורנפלקס עם חלב סויה וצימוקים": "Bowl of cornflakes with soy milk and raisins",
    "תפו\"א מבושלים שעועית ירוקה וקצת קינואה, 50 גרם עוף": "Boiled potatoes, green beans, quinoa, 50g chicken",
    # המשך מילון...
}

# מילון ערכים תזונתיים (ליחידה אחת)
nutrition_data = {
    "פיתה": {"proteins": 6, "fats": 1, "carbohydrates": 30, "dietaryFiber": 2},
    "חמאת בוטנים": {"proteins": 4, "fats": 8, "carbohydrates": 3, "dietaryFiber": 1},
    "קפה": {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 2, "fats": 1, "carbohydrates": 20, "dietaryFiber": 1},
    "חלב סויה": {"proteins": 3, "fats": 2, "carbohydrates": 4, "dietaryFiber": 1},
    "צימוקים": {"proteins": 1, "fats": 0, "carbohydrates": 10, "dietaryFiber": 1},
    "תפוח אדמה": {"proteins": 2, "fats": 0, "carbohydrates": 17, "dietaryFiber": 2},
    "שעועית ירוקה": {"proteins": 2, "fats": 0, "carbohydrates": 5, "dietaryFiber": 3},
    "קינואה": {"proteins": 4, "fats": 2, "carbohydrates": 15, "dietaryFiber": 2},
    "עוף": {"proteins": 15, "fats": 3, "carbohydrates": 0, "dietaryFiber": 0},
    # המשך...
}

def extract_nutrition(food_name):
    # ניקוי מילים כלליות
    food_name = food_name.replace("קערת", "").replace("צלחת", "").replace("חצי", "0.5").replace("רבע", "0.25")
    items = [item.strip() for item in food_name.replace("עם", ",").split(",")]
    totals = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    
    for item in items:
        quantity = 1
        for word in item.split():
            try:
                quantity = float(word)
                item = item.replace(word, "").strip()
                break
            except:
                continue
        for key in nutrition_data:
            if key in item:
                values = nutrition_data[key]
                for k in totals:
                    totals[k] += values[k] * quantity
                break
    return totals

def translate_value(value, key=None):
    if key == "notes":
        return value

    if isinstance(value, str):
        if value in translation_cache:
            return translation_cache[value]
        if any('\u0590' <= c <= '\u05FF' for c in value):
            try:
                result = translator.translate(value, "English")
                translation_cache[value] = result.result
                return result.result
            except:
                return value
        return value
    elif isinstance(value, dict):
        new_dict = {}
        for k, v in value.items():
            if k == "nutritionalValues" and "foodName" in value:
                food = value["foodName"]
                nutrients = extract_nutrition(food)
                new_dict[k] = nutrients
            else:
                new_dict[k] = translate_value(v, k)
        return new_dict
    elif isinstance(value, list):
        return [translate_value(item) for item in value]
    else:
        return value

def translate_json(file_obj):
    if file_obj is None:
        return None, None

    try:
        content = file_obj.read().decode('utf-8')
        json_content = json.loads(content)
        translated_json = translate_value(json_content)

        # extract insights
        df = pd.json_normalize(translated_json)
        df["dateTaken"] = pd.to_datetime(df.get("dateTaken", pd.NaT))
        df["month"] = df["dateTaken"].dt.month
        df["year"] = df["dateTaken"].dt.year

        insights = []
        if "nutritionalValues.proteins" in df and "feeling.value" in df:
            grouped = df.groupby(["year", "month"])
            for (y, m), group in grouped:
                morning = group[group["dateTaken"].dt.hour < 12]
                ate_protein = morning[morning["nutritionalValues.proteins"] > 5]
                mood_protein = ate_protein["feeling.value"].mean()
                no_protein = morning[morning["nutritionalValues.proteins"] <= 1]
                mood_none = no_protein["feeling.value"].mean()
                if pd.notna(mood_protein) and pd.notna(mood_none):
                    diff = round(mood_protein - mood_none, 2)
                    if abs(diff) > 0.3:
                        insights.append(f"In {m}/{y}, eating proteins in the morning increased mood by {diff}")

        # save translated file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_json, f, ensure_ascii=False, indent=2)

        return output_path, "\n".join(insights) if insights else "No significant patterns found."

    except Exception as e:
        print(f"Error translating JSON: {e}")
        return None, "Error during processing"

with gr.Blocks() as demo:
    gr.Markdown("## 🈯 JSON Translator + Nutrition + Insights", elem_id="title")

    file_input = gr.File(label="Upload JSON file", file_types=[".json"])
    output_file = gr.File(label="Download Translated JSON with Nutrition")
    insights_box = gr.Textbox(label="🧠 Pattern Insights", lines=5)

    gr.Markdown("### Select Feelings")
    feelings_selector = gr.CheckboxGroup(
        choices=["Parkinson's State", "My Mood", "Physical State"],
        label="Choose feelings to analyze",
        value=[]
    )

    gr.Markdown("### Select Data Types")
    types_selector = gr.CheckboxGroup(
        choices=["medicines", "nutritions", "activities", "symptoms"],
        label="Choose data types to include",
        value=[]
    )

    file_input.change(fn=translate_json, inputs=file_input, outputs=[output_file, insights_box])

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
