import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def translate_json(file, selected_types):
    if file is None:
        return "No file uploaded", None

    file_path = file.name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading file: {str(e)}", None

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

    # Perform translation on the whole file
    translated = [
        recursive_translate(item) for item in data
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

    gr.HTML("""
        <style>
            #file-upload-btn {
                font-size: 12px;
                padding: 5px 10px;
                height: 40px;
                width: 150px;
                border-radius: 5px;
                margin-bottom: 10px;
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

    output_graph = gr.Plot(label="Graph of Mood and Activities")
    status_message = gr.HTML(label="Status", value="")

    def handle_upload(file, types, activity):
        # Show progress message
        status_message.update(value="Uploading and processing data... Please wait.")
        
        # Perform file translation
        data = translate_json(file, types)
        if isinstance(data, str):  # Error message in case of failure
            status_message.update(value=data)
            return None  # Don't generate the graph if there was an error
        
        # If successful, update the status message
        status_message.update(value="File uploaded and translated successfully!")
        
        # Generate the graph
        return plot_graph(data, [types], activity)

    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph, status_message])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
