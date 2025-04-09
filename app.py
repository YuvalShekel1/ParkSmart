import gradio as gr
import json
import os

def translate_json(file, selected_types):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def translate_value(val):
        if isinstance(val, str) and any("\u0590" <= ch <= "\u05EA" for ch in val):  # Detect Hebrew
            return f"translated({val})"
        return val

    def recursive_translate(obj):
        if isinstance(obj, dict):
            return {k: recursive_translate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_translate(item) for item in obj]
        else:
            return translate_value(obj)

    translated = [recursive_translate(item) for item in data if item.get("type") in selected_types]

    output_path = "translated_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    return json.dumps(translated, ensure_ascii=False, indent=2), output_path

with gr.Blocks() as demo:
    gr.Markdown("## Parkinson Analyzer (Hebrew to English)")

    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"])
        selected_types = gr.CheckboxGroup(
            ["My Mood", "Parkinson's State", "Physical State", "Medication", "Exercise", "Diet"],
            label="Select types to translate",
        )

    output_json = gr.JSON(label="Translated JSON")
    download_btn = gr.File(label="Download Translated File")

    def handle_translate(file, types):
        return translate_json(file, types)

    translate_btn = gr.Button("Translate")
    translate_btn.click(fn=handle_translate, inputs=[file_input, selected_types], outputs=[output_json, download_btn])

    # Set the app to listen on all IPs and the port provided by Render
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
