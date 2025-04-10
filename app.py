import gradio as gr
import matplotlib.pyplot as plt
import os
import json
import tempfile
from deep_translator import GoogleTranslator

# פונקציה שמציגה גרף בסיסי
def create_default_graph():
    # נתוני גרף
    x = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM']
    y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]
    
    # יצירת הגרף
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Default Graph')
    
    # שמירת הגרף כתמונה זמנית להצגה ב-Gradio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name)
    plt.close()
    return temp_file.name

# פונקציה לתרגום JSON מעברית לאנגלית
def translate_json(file_obj):
    if file_obj is None:
        return None

    try:
        # ננסה לקרוא את הקובץ כתוכן טקסטואלי
        try:
            # אם יש read(), נשתמש בו (גרסת Gradio)
            content = file_obj.read().decode('utf-8')
        except AttributeError:
            # אם אין read(), נשתמש בשם הקובץ (temp path)
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                content = f.read()

        json_content = json.loads(content)

        def translate_value(value):
            if isinstance(value, str):
                hebrew_chars = any('\u0590' <= c <= '\u05FF' for c in value)
                if hebrew_chars:
                    print(f"Translating: {value}")
                    try:
                        return GoogleTranslator(source='he', target='en').translate(value)
                    except Exception as e:
                        print(f"Translation error for text '{value}': {str(e)}")
                        return value
                return value
            elif isinstance(value, dict):
                return {k: translate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [translate_value(item) for item in value]
            else:
                return value

        translated_json = translate_value(json_content)

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_json, f, ensure_ascii=False, indent=2)

        return output_path

    except Exception as e:
        print(f"Error translating JSON: {str(e)}")
        return None


# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - JSON Translator")
    gr.Markdown("Upload a JSON file with Hebrew text to translate it to English")

    # הצגת גרף בסיסי כשהאפליקציה עולה
    graph_output = gr.Image(label="Graph", visible=True)
    
    # העלאת קובץ JSON
    file_input = gr.File(label="Upload JSON File", file_types=[".json"])
    
    result_output = gr.File(label="Translated File")
    status_output = gr.Markdown("")

    # פונקציה להעלאת ותרגום הקובץ
    def handle_upload(file):
        if file is None:
            return None, "Please upload a JSON file first"
        
        translated_file_path = translate_json(file)
        if translated_file_path:
            return translated_file_path, "✅ Translation completed! Click to download the translated file."
        else:
            return None, "❌ Error translating file. Please check the format and try again."

    translate_btn = gr.Button("Translate JSON")
    translate_btn.click(
        fn=handle_upload, 
        inputs=[file_input], 
        outputs=[result_output, status_output]
    )
    
    # הצגת הגרף בסיסי בטעינה
    demo.load(lambda: create_default_graph(), None, graph_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
