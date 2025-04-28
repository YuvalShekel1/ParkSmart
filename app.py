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

# Enhanced nutrition database with more accurate values
nutrition_db = {
    # Basic ingredients
    "פיתה": {"proteins": 6, "fats": 1.5, "carbohydrates": 33, "dietaryFiber": 1.5},
    "חצי פיתה": {"proteins": 3, "fats": 0.75, "carbohydrates": 16.5, "dietaryFiber": 0.75},
    "רבע פיתה": {"proteins": 1.5, "fats": 0.38, "carbohydrates": 8.25, "dietaryFiber": 0.38},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "ממרח בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "קפה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 0.4, "dietaryFiber": 0},
    "סלמון": {"proteins": 25, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 7, "fats": 1, "carbohydrates": 84, "dietaryFiber": 3},
    "קערת קורנפלקס": {"proteins": 7, "fats": 1, "carbohydrates": 84, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3.3, "fats": 2, "carbohydrates": 4, "dietaryFiber": 0.5},
    "חלב שקדים": {"proteins": 1.1, "fats": 2.5, "carbohydrates": 3, "dietaryFiber": 0.7},
    "צימוקים": {"proteins": 0.5, "fats": 0.2, "carbohydrates": 17, "dietaryFiber": 0.8},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "עגבנייה": {"proteins": 0.9, "fats": 0.2, "carbohydrates": 3.9, "dietaryFiber": 1.2},
    "פלפל": {"proteins": 1, "fats": 0.3, "carbohydrates": 6, "dietaryFiber": 2.1},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
    "טחינה": {"proteins": 17, "fats": 57, "carbohydrates": 10, "dietaryFiber": 10},
    "פירה": {"proteins": 2, "fats": 0.1, "carbohydrates": 15, "dietaryFiber": 1.5},
    "אפונה": {"proteins": 5, "fats": 0.4, "carbohydrates": 14, "dietaryFiber": 5},
    "שקדים": {"proteins": 21, "fats": 49, "carbohydrates": 22, "dietaryFiber": 12.5},
    "מעדן סויה": {"proteins": 3.5, "fats": 2, "carbohydrates": 15, "dietaryFiber": 1.5},
    "אפרסק": {"proteins": 0.9, "fats": 0.25, "carbohydrates": 9.5, "dietaryFiber": 1.5},
    "תפוח": {"proteins": 0.3, "fats": 0.2, "carbohydrates": 14, "dietaryFiber": 2.4},
    "עוגת תפוחים": {"proteins": 3, "fats": 13, "carbohydrates": 38, "dietaryFiber": 1.5},
    "תפו\"א": {"proteins": 2, "fats": 0.1, "carbohydrates": 17, "dietaryFiber": 2},
    "שעועית ירוקה": {"proteins": 1.8, "fats": 0.1, "carbohydrates": 7, "dietaryFiber": 2.7},
    "קינואה": {"proteins": 4, "fats": 1.8, "carbohydrates": 21, "dietaryFiber": 2.8},
    "עוף": {"proteins": 27, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "ביצים": {"proteins": 13, "fats": 11, "carbohydrates": 1, "dietaryFiber": 0},
    "סלט ביצים": {"proteins": 10, "fats": 10, "carbohydrates": 1, "dietaryFiber": 0},
    "מרק ירקות": {"proteins": 2, "fats": 1, "carbohydrates": 10, "dietaryFiber": 2},
    "פתיתים": {"proteins": 5, "fats": 1, "carbohydrates": 30, "dietaryFiber": 1.5},
    "מרק אפונה": {"proteins": 5, "fats": 1, "carbohydrates": 15, "dietaryFiber": 5},
    "כרובית": {"proteins": 2, "fats": 0.3, "carbohydrates": 5, "dietaryFiber": 2},
    "סלרי": {"proteins": 0.7, "fats": 0.2, "carbohydrates": 3, "dietaryFiber": 1.6},
    "מרק סלרי": {"proteins": 1, "fats": 0.5, "carbohydrates": 4, "dietaryFiber": 1.5},
    "ריבה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 38, "dietaryFiber": 0.5},
    "קוטג": {"proteins": 11, "fats": 4.5, "carbohydrates": 3.5, "dietaryFiber": 0},
    "פאי אגסים": {"proteins": 3, "fats": 12, "carbohydrates": 40, "dietaryFiber": 2},
    
    # Common combinations
    "חצי פיתה עם חמאת בוטנים": {"proteins": 11, "fats": 16.75, "carbohydrates": 22.5, "dietaryFiber": 2.75},
    "פלפל ומלפפון": {"proteins": 1.7, "fats": 0.4, "carbohydrates": 8.5, "dietaryFiber": 2.6},
    "קערת קורנפלקס עם חלב סויה וצימוקים": {"proteins": 10.8, "fats": 3.2, "carbohydrates": 105, "dietaryFiber": 4.3},
    "קערת קורנפלקס עם חלב שקדים וצימוקים": {"proteins": 8.6, "fats": 3.7, "carbohydrates": 104, "dietaryFiber": 4.5},
    "סלמון עם פירה ואפונה": {"proteins": 32, "fats": 14.5, "carbohydrates": 29, "dietaryFiber": 6.5},
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": {"proteins": 33.6, "fats": 27.8, "carbohydrates": 49.4, "dietaryFiber": 5.2},
    "מעדן סויה אפרסק": {"proteins": 4.4, "fats": 2.25, "carbohydrates": 24.5, "dietaryFiber": 3},
    "שקדים טבעיים": {"proteins": 21, "fats": 49, "carbohydrates": 22, "dietaryFiber": 12.5},
    "תפו\"א מבושלים שעועית ירוקה וקצת קינואה, 50 גרם עוף": {"proteins": 20.8, "fats": 5.9, "carbohydrates": 45, "dietaryFiber": 7.5},
    "תפו\"א מבושלים, סלט ביצים": {"proteins": 12, "fats": 10.1, "carbohydrates": 18, "dietaryFiber": 2},
    "מרק ירקות עם פתיתים": {"proteins": 7, "fats": 2, "carbohydrates": 40, "dietaryFiber": 3.5},
    "מרק אפונה, כרובית מבושלת": {"proteins": 7, "fats": 1.3, "carbohydrates": 20, "dietaryFiber": 7},
    "צלחת מרק סלרי": {"proteins": 1, "fats": 0.5, "carbohydrates": 4, "dietaryFiber": 1.5},
    "חצי פיתה עם ריבה": {"proteins": 3.3, "fats": 0.85, "carbohydrates": 54.5, "dietaryFiber": 1.25},
    "פלפל עם קוטג": {"proteins": 12, "fats": 4.8, "carbohydrates": 9.5, "dietaryFiber": 2.1},
    "פאי אגסים וקפה קטן": {"proteins": 3.3, "fats": 12.1, "carbohydrates": 40.4, "dietaryFiber": 2},
    "רבע פיתה עם ממרח בוטנים": {"proteins": 9.5, "fats": 16.38, "carbohydrates": 14.25, "dietaryFiber": 2.38},
}

translated_data_global = []

def extract_food_nutrition(food_name):
    """Extract nutritional values from a food item based on the nutrition database"""
    # Direct match - check if the exact food name exists in our database
    if food_name in nutrition_db:
        return nutrition_db[food_name]
    
    # Partial match - look for components in the food name
    total = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    matched = False
    
    # Sort by length descending to match longer phrases first
    food_keys = sorted(nutrition_db.keys(), key=len, reverse=True)
    
    for key in food_keys:
        if key in food_name:
            matched = True
            for nutrient in total:
                total[nutrient] += nutrition_db[key][nutrient]
    
    # If no match was found, return default values
    if not matched:
        return {"proteins": 5, "fats": 5, "carbohydrates": 15, "dietaryFiber": 2}
    
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
        return [translate_value(item) for item in item]
    else:
        return value

def process_json_file(file_path):
    """Process the uploaded JSON file, translate it, and add nutritional values"""
    global translated_data_global
    
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as a complete JSON object first
        try:
            # Parse the entire JSON content
            json_data = json.loads(content)
            
            # Determine structure and identify the nutrition data
            if isinstance(json_data, dict) and "nutritions" in json_data:
                # Case 1: JSON object with nutrition array inside
                nutritions_data = json_data["nutritions"]
                is_nested = True
            else:
                # Case 2: Direct array or object without nesting
                nutritions_data = json_data if isinstance(json_data, list) else [json_data]
                is_nested = False
                
        except json.JSONDecodeError:
            # Try to parse as fragment by wrapping appropriately
            try:
                content = content.strip()
                if content.lstrip().startswith('"') and ':' in content:
                    # Wrap as object
                    modified_content = '{' + content + '}'
                    json_data = json.loads(modified_content)
                    nutritions_data = json_data.get("nutritions", [])
                    is_nested = "nutritions" in json_data
                else:
                    # Wrap as array
                    modified_content = '[' + content + ']'
                    json_data = json.loads(modified_content)
                    nutritions_data = json_data
                    is_nested = False
            except json.JSONDecodeError:
                return False, None, "Invalid JSON format. Please check your file."
        
        # Process nutritional values in the nutrition data
        if isinstance(nutritions_data, list):
            for entry in nutritions_data:
                if isinstance(entry, dict) and "foodName" in entry:
                    # Get the Hebrew food name
                    hebrew_food_name = entry["foodName"]
                    
                    # Get accurate nutritional values
                    nutritional_values = extract_food_nutrition(hebrew_food_name)
                    
                    # Update the nutritionalValues
                    entry["nutritionalValues"] = nutritional_values
        
        # Translate the entire JSON structure
        translated_full = translate_value(json_data)
        
        # Save the nutrition data separately for analysis
        if is_nested and "nutritions" in translated_full and isinstance(translated_full["nutritions"], list):
            translated_data_global = translated_full["nutritions"]
        else:
            translated_data_global = translated_full if isinstance(translated_full, list) else [translated_full]
        
        # Save to a temporary file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_full, f, ensure_ascii=False, indent=2)
        
        # Return status and file path
        nutrition_count = len(nutritions_data) if isinstance(nutritions_data, list) else 1
        return True, output_path, f"Processing complete! Found {nutrition_count} nutrition entries. The entire file has been translated and nutritional values have been updated."
    
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
        
        # Check for date fields - try different possible field names
        date_fields = ["date", "dateTaken", "createdAt"]
        date_field = None
        
        for field in date_fields:
            if field in df.columns:
                date_field = field
                break
                
        if date_field is None:
            return "No valid date field found in the data."
            
        # Convert date strings to datetime objects
        df[date_field] = pd.to_datetime(df[date_field], errors='coerce')

        # Filter data for the selected year and month
        if not df[date_field].isna().all():  # Check if there are valid dates
            df = df[(df[date_field].dt.month == int(month)) & (df[date_field].dt.year == int(year))]
        
        # Check if the data is empty after filtering
        if len(df) == 0:
            return f"No data found for {year}-{month}."
        
        # For mood field, try different possible field names if the selected one isn't found
        mood_fields = {
            "Parkinson's State": ["Parkinson's State", "ParkinsonState", "parkinsonState"],
            "My Mood": ["My Mood", "myMood", "mood"],
            "Physical State": ["Physical State", "physicalState", "physical"]
        }
        
        # Try to find a valid mood field
        valid_mood_field = None
        if mood_field in mood_fields:
            for field in mood_fields[mood_field]:
                if field in df.columns:
                    valid_mood_field = field
                    break
        
        if valid_mood_field is None and mood_field in df.columns:
            valid_mood_field = mood_field
            
        if valid_mood_field is None:
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
        if not df[date_field].isna().all():
            df["hour"] = df[date_field].dt.hour
            df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 10, 15, 24], labels=["morning", "noon", "evening"])

        # Create insights based on time of day
        insights = "Insights for selected period:\n\n"
        
        # Analyze by time of day if there's enough data
        try:
            # Handle both numeric and string mood values
            if valid_mood_field in df.columns:
                df[valid_mood_field] = pd.to_numeric(df[valid_mood_field], errors='coerce')
            df[nutrition_field] = pd.to_numeric(df[nutrition_field], errors='coerce')
            
            # Filter out rows with missing values
            valid_data = df.dropna(subset=[valid_mood_field, nutrition_field])
            
            if len(valid_data) == 0:
                return "No valid numeric data found for analysis."
                
            if "time_of_day" in valid_data.columns:
                group = valid_data.groupby("time_of_day").agg({
                    valid_mood_field: 'mean',
                    nutrition_field: 'mean'
                })
                
                for time in group.index:
                    mood_avg = round(float(group.loc[time][valid_mood_field]), 2)
                    nut_avg = round(float(group.loc[time][nutrition_field]), 2)
                    insights += f"- During {time}: Mood avg = {mood_avg}, {nutrition_field} avg = {nut_avg}\n"
            
            # Overall statistics
            insights += f"\nOverall average {valid_mood_field}: {round(valid_data[valid_mood_field].mean(), 2)}\n"
            insights += f"Overall average {nutrition_field}: {round(valid_data[nutrition_field].mean(), 2)}\n"
            
            # Add correlation if there's enough data
            if len(valid_data) >= 5:
                correlation = valid_data[valid_mood_field].corr(valid_data[nutrition_field])
                insights += f"\nCorrelation between {valid_mood_field} and {nutrition_field}: {round(correlation, 3)}\n"
                
                if abs(correlation) > 0.5:
                    if correlation > 0:
                        insights += f"There appears to be a positive relationship between {valid_mood_field} and {nutrition_field}."
                    else:
                        insights += f"There appears to be a negative relationship between {valid_mood_field} and {nutrition_field}."
            
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
        year_selector = gr.Dropdown(choices=["2024", "2025"], label="Select Year", value="2025")
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], label="Select Month", value="2")
    
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
