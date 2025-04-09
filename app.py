import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def translate_json(file, selected_types):
    if file is None:
        return [], None  # Return empty data and None when no file is uploaded

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
    
    return translated, data  # Return translated data along with original data

def plot_graph(data, selected_types, selected_activity, symbol="☕"):
    fig, ax = plt.subplots(figsize=(10, 6))

    hours = np.arange(0, 24, 1)  # 0-23 hours
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    # Plotting empty graph initially if no data exists
    if len(data) == 0:
        plt.tight_layout()
        return fig

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

    # Add CSS specifically for the file upload button (without affecting the rest of the page)
    gr.HTML("""
        <style>
            /* Custom style for the file upload button only */
            #file-upload-btn {
                font-size: 12px;  /* Small font size */
                padding: 5px 10px;  /* Smaller padding */
                height: 40px;  /* Smaller height */
                width: 150px;  /* Smaller width */
                border-radius: 5px; /* Rounded corners for square button */
                margin-bottom: 10px;  /* Optional: Space below the button */
            }
        </style>
    """)

    # Upload JSON button with small size
    with gr.Row():
        file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    
    # Radio buttons to select feelings to visualize
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
    )

    # Radio buttons to select activity to visualize
    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
    )

    # Initial empty graph with X and Y axis only
    output_graph = gr.Plot(label="Graph of Mood and Activities")

    def handle_upload(file, types, activity):
        data, _ = translate_json(file, types)
        # Return the empty graph if no file or data is uploaded
        if not data:
            return plot_graph([], types, activity)
        return plot_graph(data, [types], activity)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
