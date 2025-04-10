import gradio as gr
import json
import tempfile
from translatepy import Translator

translator = Translator()

# Custom translation cache for Hebrew phrases
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
    "התכווצויות בכפות הרגליים": "Foot cramps",
    "התכווצויות בכפות הרגליים.": "Foot cramps",
    "אזילקט": "Azilect",
    "דופיקר": "Dopicar",
    "דופיקר 125": "Dopicar 125",
    "דופיקר 175": "Dopicar 175",
    "דופיקר 250": "Dopicar 250",
    "הליכה": "Walking",
    "הרכבת כסאות גינה": "Assembling garden chairs",
    "חצי פיתה עם חמאת בוטנים": "Half pita with peanut butter",
    "חצי פיתה עם ריבה": "Half pita with jam",
    "טאקי": "Taki (card game)",
    "טורניר טנש": "Tennis tournament",
    "טנש": "Tennis",
    "מעדן סויה אפרסק": "Peach soy pudding",
    "מרק אפונה, כרובית מבושלת": "Pea soup, cooked cauliflower",
    "מרק ירקות עם פתיתים": "Vegetable soup with ptitim",
    "נסיעה לבית שאן": "Trip to Beit She'an",
    "סיור במוזיאון גולני": "Tour in Golani museum",
    "סינמט": "Cinemat",
    "סלמון עם פירה ואפונה": "Salmon with mashed potatoes and peas",
    "עבודת גינה": "Gardening",
    "עוגת תפוחים": "Apple cake",
    "פאי אגסים וקפה קטן": "Pear pie and small coffee",
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": "Pita with tahini, cucumber, tomato and small schnitzel",
    "פלפל ומלפפון": "Pepper and cucumber",
    "פלפל עם קוטג": "Pepper with cottage cheese",
    "צלחת מרק סלרי": "Bowl of celery soup",
    "קוצב": "Pacemaker",
    "קערת קורנפלקס עם חלב סויה וצימוקים": "Bowl of cornflakes with soy milk and raisins",
    "קערת קורנפלקס עם חלב שקדים וצימוקים": "Bowl of cornflakes with almond milk and raisins",
    "קפה": "Coffee",
    "רבע פיתה עם ממרח בוטנים": "Quarter pita with peanut spread",
    "שקדים טבעיים": "Natural almonds",
    "תפו\"א מבושלים שעועית ירוקה וקצת קינואה, 50 גרם עוף": "Boiled potatoes, green beans, a bit of quinoa, 50g chicken",
    "תפו\"א מבושלים, סלט ביצים": "Boiled potatoes, egg salad"
}

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
        try:
            content = file_obj.read().decode('utf-8')
        except AttributeError:
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()

        json_content = json.loads(content)
        translated_json = translate_value(json_content)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_json, f, ensure_ascii=False, indent=2)

        return output_path

    except Exception as e:
        print(f"Error translating JSON: {e}")
        return None

with gr.Blocks() as demo:
    gr.Markdown("# 🈯 JSON Hebrew to English Translator")
    file_input = gr.File(label="Upload JSON file", file_types=[".json"])
    output_file = gr.File(label="Download Translated File")
    translate_btn = gr.Button("Translate")

    translate_btn.click(translate_json, inputs=file_input, outputs=output_file)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
