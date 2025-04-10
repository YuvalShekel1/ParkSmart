import gradio as gr
import json
import tempfile
from translatepy import Translator

translator = Translator()

def translate_value(value):
    if isinstance(value, str):
        hebrew_chars = any('֐' <= c <= '׿' for c in value)
        if hebrew_chars:
            try:
                result = translator.translate(value, "English")
                return result.result
            except Exception as e:
                print(f"Translation error for '{value}': {e}")
                return value
        return value
    elif isinstance(value, dict):
        return {k: translate_value(v) for k, v in value.items()}
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
    demo.launch()
