import gradio as gr
import json
import os
import matplotlib.pyplot as plt
import numpy as np

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

# This will generate an empty graph when the app first loads
def initial_graph():
    return plot_graph()

with gr.Blocks() as demo:
    gr.Markdown("## ParkSmart - Analyze Your Data")

    # Add CSS for layout and file upload button
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
            .gradio-plot {
                width: 100%;
                height: 600px;
            }
        </style>
    """)

    # File upload input
    file_input = gr.File(label="Upload JSON", file_types=[".json"], elem_id="file-upload-btn")
    
    # Radio buttons to select feelings and activity to visualize
    selected_types = gr.Radio(
        ["My Mood", "Parkinson's State", "Physical State"],
        label="Select feelings to visualize",
    )

    selected_activity = gr.Radio(
        ["symptoms", "medicines", "nutritions", "activities"],
        label="Select activity to visualize",
    )

    # Output area for the graph
    output_graph = gr.Plot(label="Graph of Mood and Activities")

    def handle_upload(file, types, activity):
        data, _ = translate_json(file, types)
        return plot_graph(data, [types], activity)

    # Generate button to trigger the graph update
    translate_btn = gr.Button("Generate Visualization")
    translate_btn.click(fn=handle_upload, inputs=[file_input, selected_types, selected_activity], outputs=[output_graph])

    # Initially load a blank graph with axes only
    demo.load(fn=initial_graph, outputs=[output_graph])

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
