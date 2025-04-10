import gradio as gr
import matplotlib.pyplot as plt

# פונקציה שמציגה גרף בסיסי
def create_default_graph():
    # נתוני גרף
    x = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM', '8 AM', '9 AM', '10 AM', '11 AM']
    y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]
    
    # יצירת הגרף
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Default Graph')
    
    # הצגת הגרף בלי לשמור אותו לקובץ
    plt.show()

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # הצגת גרף בסיסי כשהאפליקציה עולה
    create_default_graph()  # מציג את הגרף הבסיסי

    # העלאת קובץ JSON עם כפתור קטן
    file_input = gr.File(label="Upload JSON", file_types=[".json"])

    # פונקציה להעלאת הקובץ
    def handle_upload(file):
        file_url = translate_json(file)  # תרגום הקובץ
        return gr.HTML(f'<a href="{file_url}" target="_blank">Download Translated File</a>')  # הצגת קישור להורדה

    translate_btn = gr.Button("Generate Translated File")
    translate_btn.click(fn=handle_upload, inputs=[file_input], outputs=[gr.HTML()])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
