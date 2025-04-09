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

def plot_graph(data=None, selected_types=None, selected_activity=None, symbol="☕"):
    fig, ax = plt.subplots(figsize=(15, 8))  # Increased size for bigger graph

    # Create an empty graph with just the X and Y axes
    hours = np.arange(0, 24, 1)  # 0-23 hours
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{(i+12)%24}:00" for i in hours], rotation=45)

    ax.set_ylim(0, 5)
    ax.set_ylabel('Feelings (1-5)')

    ax.set_title("Mood, Parkinson's State & Physical State over Time")

    # Plotting empty graph initially if no data exists
    if data is None or len(data) == 0:
        ax.text(12, 3, "No Data Available", ha="center", va="center", fontsize=14, color="gray")  # Show message when no data
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

def initial_graph():
    return plot_graph()

with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # Add custom CSS for layout
    gr.HTML("""
        <style>
            /* Style for file upload button */
            #file-upload-btn {
                font-size: 12px;
                padding: 5px 10px;
                height: 40px;
                width: 150px;
                border-radius: 5px;
                margin-bottom: 10px;
            }

            /* Style for layout */
            .gradio-container {
                font-family: 'Arial', sans-serif;
                padding: 20px;
                background-color: #f4f4f9;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            /* Style for graph display */
            .gradio-plot {
                width: 100%;
                height: 600px;
                border: 2px solid #ddd;
                border-radius: 10px;
                background-color: #fff;
            }

            /* Style for making buttons side by side */
            .gradio-row {
                display: flex;
                justify-content: space-between;
            }

            .gradio-radio {
                margin-right: 15px;
            }

            .gradio-button {
                margin-top: 15px;
                margin-bottom: 10px;
            }

            /* Style for the "Upload successful" message */
            .file-upload-msg {
                font-size: 14px;
                color: green;
                margin-top: 10px;
            }

            /* Style for the loading spinner */
            .gradio-spinner {
                width: 24px;
                height: 24px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                animation: spin 2s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    """)

    # File upload input
    file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    file_message = gr.HTML("<div class='file-upload-msg'>No file uploaded yet.</div>")

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

    # Graph of Mood and Activities
    output_graph = gr.Plot(label="Graph of Mood and Activities")

    # Show loading spinner while uploading file
    def show_loading_spinner():
        return gr.HTML('<div class="gradio-spinner"></div>')

    def handle_upload(file, types, activity):
        data = translate_json(file, types)
        return plot_graph(data, [types], activity), "Upload successful!"

    # Handle file upload event
    def handle_file_upload(file):
        return show_loading_spinner(), None

    file_input.upload(fn=handle_file_upload, inputs=[file_input], outputs=[file_message, output_graph])

    # Generate button to trigger the graph update
    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph, file_message])

    # Initially load a blank graph with axes only
    demo.load(fn=initial_graph, outputs=[output_graph])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
