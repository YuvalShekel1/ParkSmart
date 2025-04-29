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
                    item["date"] = item["dateTaken"]
        
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
                elif "dateTaken" in item:
                    date = pd.to_datetime(item["dateTaken"])
                
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
    insights = "🍎 תובנות תזונה:\n"
    
    # איחוד נתונים לפי תאריך לניתוח (שימוש בקרבת זמנים)
    combined_data = []
    
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        
        # מציאת רשומות תזונה באותו יום
        same_day_nutrition = nutrition_df[nutrition_df["date"].dt.date == mood_date.date()]
        
        for _, nutr_row in same_day_nutrition.iterrows():
            nutrition_item = nutr_row["item"]
            
            # חישוב הפרש זמנים בשעות
            if "dateTaken" in nutrition_item:
                nutr_time = pd.to_datetime(nutrition_item["dateTaken"])
                time_diff = abs((mood_date - nutr_time).total_seconds() / 3600)
                
                # התייחסות רק לרשומות בטווח של 3 שעות
                if time_diff <= 3:
                    food_name = nutrition_item.get("foodName", "")
                    nutr_values = nutrition_item.get("nutritionalValues", {})
                    
                    # סיווג סוגי מזון לפי ערכים תזונתיים
                    food_type = []
                    proteins = nutr_values.get("proteins", 0)
                    carbs = nutr_values.get("carbohydrates", 0)
                    fats = nutr_values.get("fats", 0)
                    
                    if proteins >= 10:
                        food_type.append("high_protein")
                    if carbs >= 15:
                        food_type.append("high_carbs")
                    if fats >= 10:
                        food_type.append("high_fat")
                    
                    combined_data.append({
                        "mood_value": mood_value,
                        "food_name": food_name,
                        "proteins": proteins,
                        "fats": fats,
                        "carbs": carbs,
                        "fiber": nutr_values.get("dietaryFiber", 0),
                        "time_diff": time_diff,
                        "food_type": food_type
                    })
    
    if not combined_data:
        return insights + "אין מספיק נתונים עם תזמון קרוב לניתוח קורלציה בין תזונה למצב רוח.\n"
    
    # המרה ל-DataFrame לניתוח
    analysis_df = pd.DataFrame(combined_data)
    
    # ניתוח השפעת חלבון
    high_protein = analysis_df[analysis_df["proteins"] > 10]["mood_value"].mean()
    low_protein = analysis_df[analysis_df["proteins"] <= 10]["mood_value"].mean()
    
    if not np.isnan(high_protein) and not np.isnan(low_protein) and abs(high_protein - low_protein) > 0.5:
        insights += f"• בצריכת מזונות עתירי חלבון (>10 גרם), מצב הרוח ממוצע {round(high_protein, 1)} לעומת {round(low_protein, 1)} עם מזונות דלי חלבון.\n"
    
    # ניתוח השפעת פחמימות
    high_carbs = analysis_df[analysis_df["carbs"] > 15]["mood_value"].mean()
    low_carbs = analysis_df[analysis_df["carbs"] <= 15]["mood_value"].mean()
    
    if not np.isnan(high_carbs) and not np.isnan(low_carbs) and abs(high_carbs - low_carbs) > 0.5:
        better = "משתפר" if high_carbs > low_carbs else "מתדרדר"
        insights += f"• אחרי אכילת פחמימות גבוהות (>15 גרם), מצב הרוח בדרך כלל {better} לציון {round(high_carbs, 1)}.\n"
    
    # ניתוח השפעת שומן
    high_fat = analysis_df[analysis_df["fats"] > 10]["mood_value"].mean()
    low_fat = analysis_df[analysis_df["fats"] <= 10]["mood_value"].mean()
    
    if not np.isnan(high_fat) and not np.isnan(low_fat) and abs(high_fat - low_fat) > 0.5:
        better = "משתפר" if high_fat > low_fat else "מתדרדר"
        insights += f"• אחרי אכילת מזונות עתירי שומן (>10 גרם), מצב הרוח בדרך כלל {better} לציון {round(high_fat, 1)}.\n"
    
    # ניתוח מזונות ספציפיים שחוזרים על עצמם
    food_mood_avg = {}
    for food in set(analysis_df["food_name"]):
        if food:  # רק אם שם המזון לא ריק
            food_data = analysis_df[analysis_df["food_name"] == food]
            if len(food_data) >= 2:  # לפחות 2 פעמים שהמזון הזה נאכל
                avg_mood = food_data["mood_value"].mean()
                food_mood_avg[food] = (avg_mood, len(food_data))
    
    # מציאת מזונות עם השפעה חזקה על מצב הרוח
    significant_foods = []
    avg_mood = analysis_df["mood_value"].mean()
    
    for food, (mood, count) in food_mood_avg.items():
        if abs(mood - avg_mood) > 0.7 and count >= 2:
            effect = "משפר" if mood > avg_mood else "מוריד"
            significant_foods.append((food, mood, effect, count))
    
    if significant_foods:
        insights += "\n• מזונות עם השפעה משמעותית:\n"
        for food, mood, effect, count in sorted(significant_foods, key=lambda x: abs(x[1]-avg_mood), reverse=True):
            insights += f"  - {food} {effect} את מצב הרוח לערך {round(mood, 1)} (נצפה {count} פעמים)\n"
    
    # מציאת שילובי סוגי מזון
    type_combinations = []
    for idx, row in analysis_df.iterrows():
        types = row["food_type"]
        mood = row["mood_value"]
        if types:
            type_string = "+".join(sorted(types))
            type_combinations.append((type_string, mood))
    
    # ניתוח השפעות שילובי סוגי מזון
    type_mood = {}
    for combo, mood in type_combinations:
        if combo not in type_mood:
            type_mood[combo] = []
        type_mood[combo].append(mood)
    
    # הצגת תובנות על שילובי מזון משמעותיים
    significant_combos = []
    for combo, moods in type_mood.items():
        if len(moods) >= 2:  # לפחות 2 פעמים שהשילוב הזה נאכל
            avg_combo_mood = sum(moods) / len(moods)
            if abs(avg_combo_mood - avg_mood) > 0.5:
                effect = "משפר" if avg_combo_mood > avg_mood else "מוריד"
                combo_name = combo.replace("high_protein", "חלבון גבוה").replace("high_carbs", "פחמימות גבוהות").replace("high_fat", "שומן גבוה")
                significant_combos.append((combo_name, avg_combo_mood, effect, len(moods)))
    
    if significant_combos:
        insights += "\n• שילובי מזון עם השפעה משמעותית:\n"
        for combo, mood, effect, count in sorted(significant_combos, key=lambda x: abs(x[1]-avg_mood), reverse=True):
            insights += f"  - ארוחות המשלבות {combo} {effect} את מצב הרוח לערך {round(mood, 1)} (נצפה {count} פעמים)\n"
    
    # ניתוח מזונות בוקר
    morning_foods = Counter([row["food_name"] for _, row in nutrition_df.iterrows() 
                             if row["date"].hour < 10 and row["item"].get("foodName", "")]).most_common(3)
    
    if morning_foods:
        insights += "\n• מזונות בוקר נפוצים: " + ", ".join([f"{name} ({count})" for name, count in morning_foods]) + "\n"
    
    # ניתוח דפוסים לפי זמני יום
    try:
        morning_mood = analysis_df[analysis_df["date"].dt.hour < 12]["mood_value"].mean()
        afternoon_mood = analysis_df[(analysis_df["date"].dt.hour >= 12) & (analysis_df["date"].dt.hour < 18)]["mood_value"].mean()
        evening_mood = analysis_df[analysis_df["date"].dt.hour >= 18]["mood_value"].mean()
        
        best_time = ""
        if not np.isnan(morning_mood) and not np.isnan(afternoon_mood) and not np.isnan(evening_mood):
            times = {"בוקר": morning_mood, "צהריים": afternoon_mood, "ערב": evening_mood}
            best_time = max(times, key=times.get)
            worst_time = min(times, key=times.get)
            
            if abs(times[best_time] - times[worst_time]) > 0.5:
                insights += f"\n• מצב הרוח נוטה להיות טוב יותר אחרי ארוחות {best_time} (ממוצע {round(times[best_time], 1)}).\n"
    except:
        pass
    
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
    gr.Markdown("## 🗣️
