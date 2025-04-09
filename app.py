import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def translate_json(file, selected_types):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def translate_value(val):
        if isinstance(val, str) and any("\u0590" <= ch <= "\u05EA" for ch in val):  # Detect Hebrew
            return f"translated({val})"  # Example translation
        return val

    def recursive_translate(obj):
        if isinstance(obj, dict):
            return {k: recursive_translate(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_translate(item) for item in obj]
        else:
            return translate_value(obj)

    translated = [
        recursive_translate(item) for item in data if item.get("type") in selected_types
    ]

    return translated

def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    fig, ax = plt.subplots(figsize=(10, 6))

    hours = np.arange(0, 24, 1)  # 0-23 hours
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    for entry in data:
        hour = entry.get('hour', 0)
        value = entry.get('value', 3)

        if selected_activity == 'nutritions' and hour == entry.get("hour"):
            ax.text(hour, value, symbol, fontsize=12, color="blue", ha='center')

        if "My Mood" in selected_types:
            ax.plot(hour, value, 'ro')
        if "Parkinson's State" in selected_types:
            ax.plot(hour, value, 'bo')
        if "Physical State" in selected_types:
            ax.plot(hour, value, 'go')

    plt.tight_layout()
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # Add CSS for small square upload button (no effect on entire page)
    gr.HTML("""
        <style>
            #file-upload-btn {
                font-size: 12px;  /* Small font size */
                padding: 5px 10px;  /* Smaller padding */
                height: 40px;  /* Smaller height */
                width: 150px;  /* Smaller width */
                border-radius: 5px; /* Rounded corners for square button */
                margin-bottom: 10px;  /* Margin at the bottom */
            }

            .gradio-container {
                max-width: 800px;  /* Limit the width of the entire form */
            }

            .disabled {
                background-color: #d3d3d3;  /* Light gray color for disabled state */
                color: #a0a0a0;  /* Disabled text color */
            }

            .active {
                background-color: white;
                color: black;
            }

            #upload-feedback {
                color: green;
                font-size: 20px;
                display: none;
            }
        </style>
    """)

    # File upload button with a square shape and smaller size
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")

    # Add a label for the success feedback message
    upload_feedback = gr.HTML('<div id="upload-feedback">✔️ File Uploaded Successfully!</div>')

    # Disable checkboxes initially
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
        interactive=False,  # Make it disabled initially
    )

    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
        interactive=False,  # Make it disabled initially
    )

    output_graph = gr.Plot(label="Graph of Mood and Activities")

    # Enable checkboxes and show success feedback after file upload
    def handle_upload(file, types, activity, state):
        if file:
            selected_types.update(interactive=True)
            selected_activity.update(interactive=True)
            upload_feedback.update(value="<div id='upload-feedback'>✔️ File Uploaded Successfully!</div>", visible=True)
        data = translate_json(file, types)
        return plot_graph(data, [types], activity)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity, gr.State()], outputs=[output_graph, upload_feedback])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
