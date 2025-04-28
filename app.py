import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os
import pandas as pd
import numpy as np
from collections import Counter

translator = Translator()

# מילון תרגום מלא
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
    "קערת קורנפלקס עם חלב שקדים וצימוקים": "Bowl of cornflakes with almond milk and raisins",
    "סלמון עם פירה ואפונה": "Salmon with mashed potatoes and peas",
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": "Pita with tahini, cucumber, tomato and small schnitzel",
    "מעדן סויה אפרסק": "Peach soy pudding",
    "פלפל עם קוטג'": "Pepper with cottage cheese",
    "רבע פיתה עם ממרח בוטנים": "Quarter pita with peanut spread",
    "תפו\"א מבושלים שעועית ירוקה וקצת קינואה, 50 גרם עוף": "Boiled potatoes, green beans and a bit of quinoa with 50g chicken",
    "תפו\"א מבושלים, סלט ביצים": "Boiled potatoes and egg salad",
    "מרק ירקות עם פתיתים": "Vegetable soup with ptitim",
    "מרק אפונה, כרובית מבושלת": "Pea soup with cooked cauliflower",
    "צלחת מרק סלרי": "Plate of celery soup",
    "פאי אגסים וקפה קטן": "Pear pie and small coffee",
    "שקדים טבעיים": "Natural almonds",
    "עוגת תפוחים": "Apple cake",
    "חלב סויה": "Soy milk",
    "חלב שקדים": "Almond milk",
    "צימוקים": "Raisins",
    "מלפפון": "Cucumber",
    "פלפל": "Pepper",
    "טחינה": "Tahini",
    "עגבנייה": "Tomato",
    "פירה": "Mashed potatoes",
    "אפונה": "Peas",
    "שניצל": "Schnitzel",
    "חצי פיתה עם ריבה": "Half pita with jam",
    "ריבה": "Jam",
    "סלט ביצים": "Egg salad",
    "ביצים": "Eggs",
    "קוטג'": "Cottage cheese",
    "שעועית ירוקה": "Green beans",
    "קינואה": "Quinoa",
    "עוף": "Chicken",
    "פתיתים": "Ptitim",
    "כרובית מבושלת": "Cooked cauliflower",
    "כרובית": "Cauliflower",
    "תפו\"א": "Potatoes",
    "תפוחי אדמה": "Potatoes",
    "מצב פרקינסון": "Parkinson's State",
    "מצב פיזי": "Physical State",
    "מצב הרוח שלי": "My Mood",
    "סימפטומים": "symptoms",
    "תרופות": "medicines",
    "תזונה": "nutritions",
    "פעילויות": "activities",
}

# מילון ערכים תזונתיים מורחב
nutrition_db = {
    "פיתה": {"proteins": 6, "fats": 1.5, "carbohydrates": 33, "dietaryFiber": 1.5},
    "חמאת בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "ממרח בוטנים": {"proteins": 8, "fats": 16, "carbohydrates": 6, "dietaryFiber": 2},
    "קפה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 0.4, "dietaryFiber": 0},
    "סלמון": {"proteins": 25, "fats": 14, "carbohydrates": 0, "dietaryFiber": 0},
    "קורנפלקס": {"proteins": 7, "fats": 1, "carbohydrates": 84, "dietaryFiber": 3},
    "חלב סויה": {"proteins": 3.3, "fats": 2, "carbohydrates": 4, "dietaryFiber": 0.5},
    "חלב שקדים": {"proteins": 1.1, "fats": 2.5, "carbohydrates": 3, "dietaryFiber": 0.7},
    "צימוקים": {"proteins": 0.5, "fats": 0.2, "carbohydrates": 17, "dietaryFiber": 0.8},
    "מלפפון": {"proteins": 0.7, "fats": 0.1, "carbohydrates": 2.5, "dietaryFiber": 0.5},
    "פלפל": {"proteins": 1, "fats": 0.3, "carbohydrates": 6, "dietaryFiber": 2.1},
    "שניצל": {"proteins": 18, "fats": 13, "carbohydrates": 8, "dietaryFiber": 0.5},
    "טחינה": {"proteins": 17, "fats": 57, "carbohydrates": 10, "dietaryFiber": 10},
    "עגבנייה": {"proteins": 0.9, "fats": 0.2, "carbohydrates": 3.9, "dietaryFiber": 1.2},
    "פירה": {"proteins": 2, "fats": 0.1, "carbohydrates": 15, "dietaryFiber": 1.5},
    "אפונה": {"proteins": 5, "fats": 0.4, "carbohydrates": 14, "dietaryFiber": 5},
    "מעדן סויה אפרסק": {"proteins": 4.4, "fats": 2.25, "carbohydrates": 24.5, "dietaryFiber": 3},
    "שקדים טבעיים": {"proteins": 21, "fats": 49, "carbohydrates": 22, "dietaryFiber": 12.5},
    "פאי אגסים וקפה קטן": {"proteins": 3.3, "fats": 12.1, "carbohydrates": 40.4, "dietaryFiber": 2},
    "עוגת תפוחים": {"proteins": 3, "fats": 13, "carbohydrates": 38, "dietaryFiber": 1.5},
    "צלחת מרק סלרי": {"proteins": 1, "fats": 0.5, "carbohydrates": 4, "dietaryFiber": 1.5},
    "קוטג'": {"proteins": 11, "fats": 4.3, "carbohydrates": 3.5, "dietaryFiber": 0},
    "ריבה": {"proteins": 0.3, "fats": 0.1, "carbohydrates": 65, "dietaryFiber": 0.5},
    "סלט ביצים": {"proteins": 13, "fats": 10, "carbohydrates": 1, "dietaryFiber": 0},
    "ביצים": {"proteins": 13, "fats": 10, "carbohydrates": 1, "dietaryFiber": 0},
    "שעועית ירוקה": {"proteins": 1.8, "fats": 0.1, "carbohydrates": 7, "dietaryFiber": 3.4},
    "קינואה": {"proteins": 4.4, "fats": 1.9, "carbohydrates": 21.3, "dietaryFiber": 2.8},
    "עוף": {"proteins": 26.5, "fats": 3.6, "carbohydrates": 0, "dietaryFiber": 0},
    "פתיתים": {"proteins": 5, "fats": 1, "carbohydrates": 30, "dietaryFiber": 1.2},
    "כרובית מבושלת": {"proteins": 2, "fats": 0.3, "carbohydrates": 5, "dietaryFiber": 2.5},
    "כרובית": {"proteins": 2, "fats": 0.3, "carbohydrates": 5, "dietaryFiber": 2.5},
    "תפו\"א": {"proteins": 2, "fats": 0.1, "carbohydrates": 15, "dietaryFiber": 1.5},
    "תפוחי אדמה": {"proteins": 2, "fats": 0.1, "carbohydrates": 15, "dietaryFiber": 1.5},
    "מרק ירקות": {"proteins": 1.5, "fats": 0.5, "carbohydrates": 8, "dietaryFiber": 2},
    "מרק אפונה": {"proteins": 5, "fats": 1, "carbohydrates": 15, "dietaryFiber": 5},
}

# Complex meal nutrition calculator
def calculate_complex_meal_nutrition(meal_name):
    # Default values
    nutrition = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    
    # Check if we have exact match in the database
    if meal_name in nutrition_db:
        return nutrition_db[meal_name]
    
    # Split the meal into components
    components = []
    for food in nutrition_db.keys():
        if food in meal_name:
            components.append(food)
    
    # If no components found, return default values
    if not components:
        return nutrition
    
    # Calculate the nutrition by summing components
    # For complex meals, we'll adjust portions
    for component in components:
        if "חצי" in meal_name and component == "פיתה":
            # Half pita
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.5
        elif "רבע" in meal_name and component == "פיתה":
            # Quarter pita
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.25
        elif "50 גרם" in meal_name and component == "עוף":
            # 50g of chicken (about half portion)
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.5
        elif "קטן" in meal_name and component == "שניצל":
            # Small schnitzel
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.7
        elif "קערת" in meal_name and component == "קורנפלקס":
            # Bowl of cornflakes
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.6  # Portion size adjustment
        elif component in meal_name:
            # Regular component
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.8  # Small adjustment for combined foods
    
    # Round values to one decimal place
    for nutrient in nutrition:
        nutrition[nutrient] = round(nutrition[nutrient], 1)
    
    return nutrition

# פונקציות עיבוד
translated_data_global = {}
original_full_json = {}

def translate_value(value, key=None):
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
        return [translate_value(item) for item in value]  # Fixed: changed "item" to "value"
    else:
        return value

def extract_food_nutrition(food_name):
    # First check for exact match
    if food_name in nutrition_db:
        return nutrition_db[food_name]
    # Use the complex meal calculator
    return calculate_complex_meal_nutrition(food_name)

def upload_and_process(file_obj):
    global translated_data_global, original_full_json
    try:
        file_path = file_obj.name
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_full_json = json.loads(content)
        
        # Convert date fields to standard format if they exist
        if "nutritions" in original_full_json:
            for item in original_full_json["nutritions"]:
                if "dateTaken" in item:
                    item["date"] = item.pop("dateTaken")
        
        keys_to_update = ["nutritions", "activities", "medications", "symptoms"]

        for key in keys_to_update:
            if key in original_full_json:
                section = original_full_json[key]
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict):
                            # Update nutritional values for food items
                            if key == "nutritions" and "foodName" in item:
                                food_name = item["foodName"]
                                item["nutritionalValues"] = extract_food_nutrition(food_name)
                
                # Translate the section
                original_full_json[key] = translate_value(section)

        translated_data_global = original_full_json

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)

        return output_path, "✅ File processed successfully! All nutritional values have been updated."
    except Exception as e:
        return None, f"❌ Error processing: {str(e)}"

def generate_insights(year, month, mood_field, selected_category):
    if not translated_data_global:
        return "Please upload a file first."
    try:
        mood_data = []
        category_data = []
        
        # Find mood data (assuming it's in symptoms)
        if "symptoms" in translated_data_global:
            for item in translated_data_global["symptoms"]:
                if "date" in item and mood_field in item:
                    date = pd.to_datetime(item["date"])
                    if date.year == int(year) and date.month == int(month):
                        mood_data.append({
                            "date": date,
                            "value": item.get(mood_field, 0)
                        })
        
        # Get category data
        if selected_category in translated_data_global:
            for item in translated_data_global[selected_category]:
                date = None
                if "date" in item:
                    date = pd.to_datetime(item["date"])
                
                if date and date.year == int(year) and date.month == int(month):
                    category_data.append({
                        "date": date,
                        "item": item
                    })
        
        if not mood_data or not category_data:
            return f"No sufficient data for the selected period ({month}/{year}) or categories."
        
        # Create DataFrames
        mood_df = pd.DataFrame(mood_data)
        category_df = pd.DataFrame(category_data)
        
        # Generate insights based on the category
        insights = f"📊 Analysis for {selected_category} in {month}/{year}:\n\n"
        
        # Basic stats
        insights += f"Found {len(mood_data)} mood entries and {len(category_data)} {selected_category} entries.\n"
        if mood_df["value"].any():
            insights += f"Average {mood_field}: {round(mood_df['value'].mean(), 2)}\n\n"
        
        # Category-specific insights
        if selected_category == "nutritions":
            insights += generate_nutrition_insights(category_df, mood_df)
        elif selected_category == "activities":
            insights += generate_activity_insights(category_df, mood_df)
        elif selected_category == "medications":
            insights += generate_medication_insights(category_df, mood_df)
        else:
            insights += generate_symptom_insights(category_df, mood_df)
        
        return insights
    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

def generate_nutrition_insights(nutrition_df, mood_df):
    insights = "🍎 Nutrition Insights:\n"
    
    # Merge data by date for analysis (using date proximity)
    combined_data = []
    
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        
        # Find nutrition entries on the same day
        same_day_nutrition = nutrition_df[nutrition_df["date"].dt.date == mood_date.date()]
        
        for _, nutr_row in same_day_nutrition.iterrows():
            nutrition_item = nutr_row["item"]
            
            # Get time difference in hours
            if "dateTaken" in nutrition_item:
                nutr_time = pd.to_datetime(nutrition_item["dateTaken"])
                time_diff = abs((mood_date - nutr_time).total_seconds() / 3600)
                
                # Only consider entries within 3 hours
                if time_diff <= 3:
                    combined_data.append({
                        "mood_value": mood_value,
                        "food_name": nutrition_item.get("foodName", ""),
                        "proteins": nutrition_item.get("nutritionalValues", {}).get("proteins", 0),
                        "fats": nutrition_item.get("nutritionalValues", {}).get("fats", 0),
                        "carbs": nutrition_item.get("nutritionalValues", {}).get("carbohydrates", 0),
                        "fiber": nutrition_item.get("nutritionalValues", {}).get("dietaryFiber", 0),
                        "time_diff": time_diff
                    })
    
    if not combined_data:
        return insights + "Not enough close-timing data to analyze correlation between nutrition and mood.\n"
    
    # Convert to DataFrame for analysis
    analysis_df = pd.DataFrame(combined_data)
    
    # Analyze protein impact
    high_protein = analysis_df[analysis_df["proteins"] > 10]["mood_value"].mean()
    low_protein = analysis_df[analysis_df["proteins"] <= 10]["mood_value"].mean()
    
    if not np.isnan(high_protein) and not np.isnan(low_protein) and abs(high_protein - low_protein) > 0.5:
        insights += f"• When consuming high-protein foods (>10g), mood averages {round(high_protein, 1)} compared to {round(low_protein, 1)} with lower protein foods.\n"
    
    # Analyze morning nutrition
    morning_foods = Counter([row["food_name"] for _, row in nutrition_df.iterrows() 
                             if row["date"].hour < 10]).most_common(3)
    
    if morning_foods:
        insights += "• Most common morning foods: " + ", ".join([f"{name} ({count})" for name, count in morning_foods]) + "\n"
    
    # Find potential correlations
    corr_nutrients = []
    for nutrient in ["proteins", "fats", "carbs", "fiber"]:
        try:
            correlation = analysis_df[[nutrient, "mood_value"]].corr().iloc[0,1]
            if not np.isnan(correlation) and abs(correlation) > 0.3:
                corr_nutrients.append((nutrient, correlation))
        except:
            pass
    
    if corr_nutrients:
        for nutrient, corr in corr_nutrients:
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.6 else "moderate"
            insights += f"• Found {strength} {direction} correlation ({round(corr, 2)}) between {nutrient} consumption and mood.\n"
    
    # Check timing patterns
    if len(analysis_df) >= 3:
        morning_mood = analysis_df[analysis_df["time_diff"] < 1]["mood_value"].mean()  
        later_mood = analysis_df[analysis_df["time_diff"] >= 1]["mood_value"].mean()
        
        if not np.isnan(morning_mood) and not np.isnan(later_mood) and abs(morning_mood - later_mood) > 0.3:
            better_time = "soon after eating" if morning_mood > later_mood else "some time after eating"
            insights += f"• Mood tends to be better {better_time} (difference of {round(abs(morning_mood - later_mood), 1)} points).\n"
    
    return insights

def generate_activity_insights(activity_df, mood_df):
    insights = "🏃 Activity Insights:\n"
    # Add activity-specific analysis logic here
    return insights + "Activity analysis not yet implemented.\n"

def generate_medication_insights(medication_df, mood_df):
    insights = "💊 Medication Insights:\n"
    # Add medication-specific analysis logic here
    return insights + "Medication analysis not yet implemented.\n"

def generate_symptom_insights(symptom_df, mood_df):
    insights = "🩺 Symptom Insights:\n"
    # Add symptom-specific analysis logic here
    return insights + "Symptom analysis not yet implemented.\n"

# גרדייו

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🗣️ JSON Translator + Full Nutrition Update")

    with gr.Row():
        file_input = gr.File(label="⬆️ Upload your JSON file", file_types=[".json"])
        output_file = gr.File(label="⬇️ Download updated JSON")
    status_message = gr.Textbox(label="Status", interactive=False)
    file_input.change(fn=upload_and_process, inputs=file_input, outputs=[output_file, status_message])

    gr.Markdown("---")
    gr.Markdown("## 📊 Analyze by Category")

    with gr.Row():
        year_selector = gr.Dropdown(choices=["2024", "2025"], value="2025", label="Select Year")
        month_selector = gr.Dropdown(choices=[str(i) for i in range(1, 13)], value="2", label="Select Month")
    
    with gr.Row():
        mood_dropdown = gr.Dropdown(
            choices=["Parkinson's State", "My Mood", "Physical State"], 
            value="Parkinson's State",
            label="Select Mood/State Field"
        )
        category_dropdown = gr.Dropdown(
            choices=["nutritions", "activities", "medications", "symptoms"], 
            value="nutritions",
            label="Select Data Category"
        )

    insights_output = gr.Textbox(label="📊 Insights", lines=10)
    analyze_btn = gr.Button("🔍 Generate Insights")

    analyze_btn.click(
        fn=generate_insights, 
        inputs=[year_selector, month_selector, mood_dropdown, category_dropdown], 
        outputs=insights_output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
