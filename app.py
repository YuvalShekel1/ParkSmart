import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os
import pandas as pd

translator = Translator()

# Dictionary for translation
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
    "אזילקט": "Azilect",
    "דופיקר": "Dopicar",
    "דופיקר 125": "Dopicar 125",
    "דופיקר 175": "Dopicar 175",
    "דופיקר 250": "Dopicar 250",
    "קפה": "Coffee",
    "חצי פיתה עם חמאת בוטנים": "Half pita with peanut butter",
    "פלפל ומלפפון": "Pepper and cucumber",
    "קערת קורנפלקס עם חלב סויה וצימוקים": "Bowl of cornflakes with soy milk and raisins",
    "סלמון עם פירה ואפונה": "Salmon with mashed potatoes and peas",
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": "Pita with tahini, cucumber, tomato and schnitzel",
}

# Nutrition database
nutrition_db = {
    "פיתה": {"proteins": 6, "fats": 1.5, "carbohydrates": 33, "dietaryFiber": 1.5},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "קפה": {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0},
    "סלמון": {"proteins": 25, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 8, "fats": 1, "carbohydrates": 83, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3, "fats": 1.5, "carbohydrates": 2, "dietaryFiber": 0.5},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "פלפל": {"proteins": 1, "fats": 0.2, "carbohydrates": 6, "dietaryFiber": 2},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
}

translated_data_global = []

def extract_food_nutrition(food_name):
    """Extract nutritional values from a food item based on the nutrition database"""
    total = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    
    # Check against Hebrew food names in the database
    for key in nutrition_db:
        if key in food_name:
            for k in total:
                total[k] += nutrition_db[key][k]
                
    return total

def translate_value(value, key=None):
    """Translate a value from Hebrew to English"""
    if key == "notes":
        return value
    if isinstance(value, str):
        if value in translation_cache:
            return translation_cache[value]
        hebrew_chars = any('\u0590' <= c <= '\u05FF' for c in value)
        if hebrew_chars:
            try:
                result = translator.translate(value, "English")
                translation_cache[value] = result.result
                return result.result
            except:
                return value
        return value
    elif isinstance(value, dict):
        return {k: translate_value(v, k) for k, v in value.items()}
    elif isinstance(value, list):
        return [translate_value(item) for item in value]
    else:
        return value

def process_json_file(file_path):
    """Process the uploaded JSON file, translate it, and add nutritional values"""
    global translated_data_global
    
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        json_data = json.loads(content)
        
        # Store original data for reference
        original_data = json_data.copy()
        
        # Translate all data
        translated_data = translate_value(json_data)
        
        # Process each entry to add/update nutritional values
        for i, entry in enumerate(translated_data):
            if isinstance(entry, dict):
                # Initialize nutritional values if not present
                if "nutritionalValues" not in entry:
                    entry["nutritionalValues"] = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
                
                # Update nutritional values if foodName is present
                if "foodName" in entry and i < len(original_data):
                    orig_entry = original_data[i]
                    if isinstance(orig_entry, dict) and "foodName" in orig_entry:
                        orig_food_name = orig_entry.get("foodName", "")
                        values = extract_food_nutrition(orig_food_name)
                        entry["nutritionalValues"] = values
        
        # Store the translated data globally
        translated_data_global = translated_data
        
        # Save to a temporary file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        # Return status and file path
        return True, output_path, f"Translation complete! Found {len(translated_data)} entries. You can download the updated file."
    
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return False, None, f"Error: {str(e)}"

def upload_and_process(file_obj):
    """Handle file upload and processing"""
    if file_obj is None:
        return None, "Please upload a JSON file."
    
    try:
        # Use the file path property instead of trying to read the file object
        file_path = file_obj.name
        success, file_path, message = process_json_file(file_path)
        
        if success:
            return file_path, message
        else:
            return None, message
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def generate_insights(year, month, mood_field, nutrition_field):
    """Generate insights based on the translated data"""
    if not translated_data_global:
        return "Please upload a file first."

    try:
        df = pd.DataFrame(translated_data_global)
        
        # Check if date field exists
        if "date" not in df.columns:
            return "Date field not found in the data."
            
        # Convert date strings to datetime objects
        df["date"] = pd.to_datetime(df["date"], errors='coerce')

        # Filter data for the selected year and month
        if not df["date"].isna().all():  # Check if there are valid dates
            df = df[(df["date"].dt.month == int(month)) & (df["date"].dt.year == int(year))]
        
        # Check if the data is empty after filtering
        if len(df) == 0:
            return f"No data found for {year}-{month}."
        
        # Check if the mood field exists
        if mood_field not in df.columns:
            return f"Mood field '{mood_field}' not found in data."
        
        # Extract nutritional values
        if "nutritionalValues" in df.columns:
            for row_idx, row in df.iterrows():
                if isinstance(row["nutritionalValues"], dict):
                    for key in ["proteins", "fats", "carbohydrates", "dietaryFiber"]:
                        if key not in df.columns:
                            df[key] = 0
                        df.at[row_idx, key] = row["nutritionalValues"].get(key, 0)
        
        # Check if nutrition field exists
        if nutrition_field not in df.columns:
            return f"Nutrition field '{nutrition_field}' not found in data."
        
        # Add time of day
        if not df["date"].isna().all():
            df["hour"] = df["date"].dt.hour
            df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 10, 15, 24], labels=["morning", "noon", "evening"])

        # Create insights based on time of day
        insights = "Insights for selected period:\n\n"
        
        # Analyze by time of day if there's enough data
        try:
            # Handle both numeric and string mood values
            df[mood_field] = pd.to_numeric(df[mood_field], errors='coerce')
            df[nutrition_field] = pd.to_numeric(df[nutrition_field], errors='coerce')
            
            # Filter out rows with missing values
            valid_data = df.dropna(subset=[mood_field, nutrition_field])
            
            if len(valid_data) == 0:
                return "No valid numeric data found for analysis."
                
            if "time_of_day" in valid_data.columns:
                group = valid_data.groupby("time_of_day").agg({
                    mood_field: 'mean',
                    nutrition_field: 'mean'
                })
                
                for time in group.index:
                    mood_avg = round(float(group.loc[time][mood_field]), 2)
                    nut_avg = round(float(group.loc[time][nutrition_field]), 2)
                    insights += f"- During {time}: Mood avg = {mood_avg}, {nutrition_field} avg = {nut_avg}\n"
            
            # Overall statistics
            insights += f"\nOverall average {mood_field}: {round(valid_data[mood_field].mean(), 2)}\n"
            insights += f"Overall average {nutrition_field}: {round(valid_data[nutrition_field].mean(), 2)}\n"
            
            # Add correlation if there's enough data
            if len(valid_data) >= 5:
                correlation = valid_data[mood_field].corr(valid_data[nutrition_field])
                insights += f"\nCorrelation between {mood_field} and {nutrition_field}: {round(correlation, 3)}\n"
                
                if abs(correlation) > 0.5:
                    if correlation > 0:
                        insights += f"There appears to be a positive relationship between {mood_field} and {nutrition_field}."
                    else:
                        insights += f"There appears to be a negative relationship between {mood_field} and {nutrition_field}."
            
            return insights
        except Exception as e:
            return f"Error in analysis: {str(e)}\n\nPlease check if there's enough data for analysis."
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🈯 JSON Translator + Nutrition Updater")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        output_file = gr.File(label="⬇️ Download the updated file")
    
    # Translation section
    gr.Markdown("---")
    gr.Markdown("## 📅 Analyze Mood and Nutrition by Year and Month")
    
    with gr.Row():
        year_selector = gr.Dropdown(choices=["2024", "2025"], label="Select Year", value="2024")
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], label="Select Month", value="1")
    
    with gr.Row():
        mood_dropdown = gr.Dropdown(
            choices=["Parkinson's State", "My Mood", "Physical State"], 
            label="Select Mood Field",
            value="My Mood"
        )
        nutrition_dropdown = gr.Dropdown(
            choices=["proteins", "fats", "carbohydrates", "dietaryFiber"],
            label="Select Nutrition Field",
            value="proteins"
        )
    
    insights_output = gr.Textbox(label="📌 Insights", lines=8)
    analyze_btn = gr.Button("🔍 Generate Insights")
    
    # Connect the functions to the interface
    file_input.change(
        fn=upload_and_process,
        inputs=file_input, 
        outputs=[output_file, status_text]
    )
    
    analyze_btn.click(
        fn=generate_insights,
        inputs=[year_selector, month_selector, mood_dropdown, nutrition_dropdown],
        outputs=insights_output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
