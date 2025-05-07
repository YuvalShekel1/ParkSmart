import gradio as gr
import json
import tempfile
from translatepy import Translator
from datetime import datetime
import os
import pandas as pd
import numpy as np
from collections import Counter
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# סתימת אזהרות
warnings.filterwarnings('ignore')

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
    "תרופה": "medicine",
    "פעילות": "activity",
    "פיל": "Pill",
    "כדור": "Pill",
    "גבוה": "High",
    "בינוני": "Moderate",
    "נמוך": "Low",
    "רעד": "Tremor",
    "קושי בדיבור": "Speech Difficulty",
    "קשיחות": "Stiffness",
    "איטיות בתנועה": "Slowness of Movement",
    "בעיות שיווי משקל": "Balance Problems",
    "עייפות": "Fatigue",
    "כאבים": "Pain",
    "הליכה": "Walking",
    "ריצה": "Running",
    "שחייה": "Swimming",
    "יוגה": "Yoga",
    "אימון כוח": "Strength Training",
    "אימון טנש": "Tennis Training",
    "אימון טנש קבוצתי": "Group Tennis Training",
    "משעה 2020 3 משחקים. הפסקה של 15 דקות לפני המשחקים": "From 8:20 PM, 3 games. 15-minute break before the games",
    "טנש": "Tennis",
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

# חישוב ערכים תזונתיים לארוחות מורכבות
def calculate_complex_meal_nutrition(meal_name):
    # ערכים דיפולטיביים
    nutrition = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    
    # בדיקה אם יש התאמה מדויקת במסד הנתונים
    if meal_name in nutrition_db:
        return nutrition_db[meal_name]
    
    # פיצול הארוחה למרכיבים
    components = []
    for food in nutrition_db.keys():
        if food in meal_name:
            components.append(food)
    
    # אם לא נמצאו מרכיבים, החזר ערכים דיפולטיביים
    if not components:
        return nutrition
    
    # חישוב הערכים התזונתיים על ידי סכימת המרכיבים
    # עבור ארוחות מורכבות, נתאים את המנות
    for component in components:
        if "חצי" in meal_name and component == "פיתה":
            # חצי פיתה
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.5
        elif "רבע" in meal_name and component == "פיתה":
            # רבע פיתה
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.25
        elif "50 גרם" in meal_name and component == "עוף":
            # 50 גרם עוף (בערך חצי מנה)
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.5
        elif "קטן" in meal_name and component == "שניצל":
            # שניצל קטן
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.7
        elif "קערת" in meal_name and component == "קורנפלקס":
            # קערת קורנפלקס
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.6  # התאמת גודל המנה
        elif component in meal_name:
            # מרכיב רגיל
            for nutrient in nutrition:
                nutrition[nutrient] += nutrition_db[component][nutrient] * 0.8  # התאמה קטנה עבור מזונות משולבים
    
    # עיגול ערכים לספרה אחת אחרי הנקודה
    for nutrient in nutrition:
        nutrition[nutrient] = round(nutrition[nutrient], 1)
    
    return nutrition

# פונקציות עיבוד
translated_data_global = {}
original_full_json = {}

def translate_value(value, key=None):
    if key == "notes" and not value:
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

def extract_food_nutrition(food_name):
    # קודם בדוק התאמה מדויקת
    if food_name in nutrition_db:
        return nutrition_db[food_name]
    # השתמש במחשבון ארוחות מורכבות
    return calculate_complex_meal_nutrition(food_name)

def upload_and_process(file_obj):
    global translated_data_global, original_full_json
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            content = f.read()

        original_full_json = json.loads(content)
        
        # וודא שכל המפתחות הנדרשים קיימים, כולל שדה 'feelings'
        keys_to_check = ["nutritions", "activities", "medications", "symptoms", "medicines", "feelings"]
        for key in keys_to_check:
            if key not in original_full_json:
                original_full_json[key] = []
        
        # מזג 'feelings' לתוך 'symptoms' אם צריך
        if "feelings" in original_full_json and isinstance(original_full_json["feelings"], list):
            # הוסף כל תחושה לסימפטומים באותו מבנה
            for feeling in original_full_json["feelings"]:
                if isinstance(feeling, dict):
                    # כבר בפורמט הנכון
                    original_full_json["symptoms"].append(feeling)
                elif isinstance(feeling, list):
                    # רשימה של תחושות
                    original_full_json["symptoms"].extend(feeling)
        
        # המר שדות תאריך לפורמט סטנדרטי אם הם קיימים
        date_keys = ["nutritions", "activities", "medications", "symptoms", "medicines"]
        for key in date_keys:
            if key in original_full_json:
                for item in original_full_json[key]:
                    if "dateTaken" in item:
                        item["date"] = item["dateTaken"]
                    # וודא שתאריכים בפורמט עקבי
                    if "date" in item:
                        try:
                            item["date"] = pd.to_datetime(item["date"]).isoformat()
                        except:
                            pass
        
        # וודא שתרופות/מטופלות עקביות
        if "medicines" in original_full_json and "medications" not in original_full_json:
            original_full_json["medications"] = original_full_json["medicines"]
        elif "medications" in original_full_json and "medicines" not in original_full_json:
            original_full_json["medicines"] = original_full_json["medications"]
        
        keys_to_update = ["nutritions", "activities", "medications", "symptoms", "medicines"]

        for key in keys_to_update:
            if key in original_full_json:
                section = original_full_json[key]
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict):
                            # עדכן ערכים תזונתיים עבור פריטי מזון
                            if key == "nutritions" and "foodName" in item:
                                food_name = item["foodName"]
                                item["nutritionalValues"] = extract_food_nutrition(food_name)
                
                # תרגם את המקטע
                original_full_json[key] = translate_value(section)

        translated_data_global = original_full_json

        # שמור את הנתונים המעובדים
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)

        return output_path, "✅ File processed successfully! All nutritional values have been updated and data has been fully translated."
    except Exception as e:
        return None, f"❌ Error processing: {str(e)}"

# --- עזר: הכנת הדאטה פריים ---

def prepare_activity_and_mood_data(data, mood_field):
    if not data or "activities" not in data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    activity_list = []
    for item in data.get("activities", []):
        if "date" in item:
            activity_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    activity_df = pd.DataFrame(activity_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return activity_df, mood_df

def prepare_medication_and_mood_data(data, mood_field):
    if not data or "medications" not in data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    medication_list = []
    for item in data.get("medications", []):
        if "date" in item:
            medication_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    medication_df = pd.DataFrame(medication_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return medication_df, mood_df

def prepare_symptom_and_mood_data(data, mood_field):
    if not data or "symptoms" not in data:
        return pd.DataFrame(), pd.DataFrame()

    symptom_list = []
    for item in data.get("symptoms", []):
        if "date" in item:
            symptom_list.append({
                "date": pd.to_datetime(item["date"]),
                "item": item
            })
    symptom_df = pd.DataFrame(symptom_list)

    mood_list = []
    for item in data.get("symptoms", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return symptom_df, mood_df

# פונקציות יצירת תובנות בסיסיות
def generate_activity_insights(activity_df, mood_df):
    insights = "🏃 Activity Insights:\n"

    if activity_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze activities and mood correlation.\n"

    combined_data = []
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        same_day_activities = activity_df[activity_df["date"].dt.date == mood_date.date()]
        for _, act_row in same_day_activities.iterrows():
            activity_item = act_row["item"]
            time_diff = 24
            if "startTime" in activity_item:
                try:
                    act_time = pd.to_datetime(activity_item["startTime"])
                    time_diff = abs((mood_date - act_time).total_seconds() / 3600)
                except:
                    pass
            if time_diff <= 6:
                activity_type = activity_item.get("activityName", "Unknown")
                duration = activity_item.get("duration", 0)
                intensity = activity_item.get("intensity", 0)
                combined_data.append({
                    "mood_value": mood_value,
                    "activity_type": activity_type,
                    "duration": duration,
                    "intensity": intensity,
                    "time_diff": time_diff
                })

    if not combined_data or len(combined_data) < 2:
        return insights + "Not enough close-timing data.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• Activity type impact on mood:\n"
    activity_types = analysis_df["activity_type"].unique()
    for activity in activity_types:
        act_mood = analysis_df[analysis_df["activity_type"] == activity]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = act_mood - overall_mood
        if not np.isnan(act_mood):
            count = len(analysis_df[analysis_df["activity_type"] == activity])
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.5:
                insights += f"  - {activity} ({count} times): Mood tends to be {abs(round(diff, 1))} points {direction} than average.\n"
            else:
                insights += f"  - {activity} ({count} times): Mood similar to average.\n"

    return insights

def generate_medication_insights(medication_df, mood_df):
    insights = "💊 Medication Insights:\n"

    if medication_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze medications.\n"

    combined_data = []
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        same_day_meds = medication_df[medication_df["date"].dt.date == mood_date.date()]
        for _, med_row in same_day_meds.iterrows():
            med_item = med_row["item"]
            time_diff = 24
            if "timeTaken" in med_item:
                try:
                    med_time = pd.to_datetime(med_item["timeTaken"])
                    time_diff = abs((mood_date - med_time).total_seconds() / 3600)
                except:
                    pass
            if time_diff <= 6:
                med_name = med_item.get("name", "Unknown")
                dosage = med_item.get("quantity", 0)
                combined_data.append({
                    "mood_value": mood_value,
                    "medication": med_name,
                    "dosage": dosage,
                    "time_diff": time_diff
                })

    if not combined_data or len(combined_data) < 2:
        return insights + "Not enough close-timing data.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• Medication impact on mood:\n"
    medications = analysis_df["medication"].unique()
    for med in medications:
        med_mood = analysis_df[analysis_df["medication"] == med]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = med_mood - overall_mood
        if not np.isnan(med_mood):
            count = len(analysis_df[analysis_df["medication"] == med])
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.3:
                insights += f"  - {med} ({count} times): Mood tends to be {abs(round(diff, 1))} points {direction}.\n"
            else:
                insights += f"  - {med} ({count} times): Mood similar to average.\n"

    return insights

def generate_symptom_insights(symptom_df, mood_df):
    insights = "🩺 Symptom Insights:\n"

    if symptom_df.empty or mood_df.empty:
        return insights + "Not enough data to analyze symptoms.\n"

    symptom_fields = set()
    for _, row in symptom_df.iterrows():
        item = row["item"]
        for key in item.keys():
            if key not in ["date", "notes", "id", "Parkinson's State", "My Mood", "Physical State", "type", "severity", "createdAt", "updatedAt", "__v", "_id", "userId"]:
                symptom_fields.add(key)
    symptom_fields = list(symptom_fields)

    # הוסף גם את הסימפטומים שמופיעים בשדה type
    for _, row in symptom_df.iterrows():
        item = row["item"]
        if "type" in item and item["type"] not in ["Parkinson's State", "My Mood", "Physical State"]:
            symptom_fields.append(item["type"])
    
    # הסר כפילויות
    symptom_fields = list(set(symptom_fields))

    if not symptom_fields:
        return insights + "No specific symptom fields detected.\n"

    insights += "• Symptom impact on mood:\n"
    date_to_mood = {row["date"].date(): row["value"] for _, row in mood_df.iterrows()}

    for symptom in symptom_fields:
        symptom_present_moods = []
        symptom_absent_moods = []
        for _, row in symptom_df.iterrows():
            date = row["date"].date()
            item = row["item"]
            if date in date_to_mood:
                mood_value = date_to_mood[date]
                symptom_present = False
                # בדוק אם הסימפטום מופיע כשדה
                if symptom in item and item[symptom]:
                    symptom_present = True
                # בדוק אם הסימפטום מופיע כערך בשדה type
                if "type" in item and item["type"] == symptom:
                    symptom_present = True
                
                if symptom_present:
                    symptom_present_moods.append(mood_value)
                else:
                    symptom_absent_moods.append(mood_value)
                    
        if symptom_present_moods and symptom_absent_moods:
            present_avg = np.mean(symptom_present_moods)
            absent_avg = np.mean(symptom_absent_moods)
            diff = present_avg - absent_avg
            direction = "higher" if diff > 0 else "lower"
            if abs(diff) >= 0.3:
                insights += f"  - {symptom}: Mood {direction} by {round(abs(diff),1)} points when present.\n"
            else:
                insights += f"  - {symptom}: No strong mood impact.\n"

    return insights

# פונקציות ניתוח מתקדמות
def analyze_activity_patterns(data, mood_field):
    if not data or "activities" not in data or "symptoms" not in data:
        return "Not enough data for activity pattern analysis."
    
    try:
        # חילוץ נתוני פעילות
        activity_data = []
        for item in data.get("activities", []):
            if "date" in item and "activityName" in item and "duration" in item and "intensity" in item:
                # וודא שהשם המדויק של הפעילות נלקח מהשדה הנכון
                activity_name = item.get("activityName", "")
                if not activity_name or len(activity_name) < 2:
                    continue  # דלג על פעילויות ללא שם תקין
                
                activity_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "name": activity_name,
                    "duration": item.get("duration", 0),
                    "intensity": item.get("intensity", "Low"),
                    "notes": item.get("notes", "")
                })
        
        # קבל נתוני מצב רוח/סימפטום
        mood_data = []
        for item in data["symptoms"]:
            if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item.get("severity", 0)
                })
        
        if len(activity_data) < 3 or len(mood_data) < 3:
            return "Not enough data points for activity analysis."
        
        # צור DataFrames
        activity_df = pd.DataFrame(activity_data)
        mood_df = pd.DataFrame(mood_data)
        
        # המר עוצמה למספרי
        intensity_map = {"Low": 1, "Moderate": 2, "High": 3}
        activity_df["intensity_score"] = activity_df["intensity"].map(lambda x: intensity_map.get(x, 1))
        
        # חשב ציון פעילות (משך * עוצמה)
        activity_df["activity_score"] = activity_df["duration"] * activity_df["intensity_score"]
        
        # התאם פעילויות עם מצב רוח (תוך 6 שעות)
        matched_data = []
        
        for _, act_row in activity_df.iterrows():
            act_date = act_row["date"]
            
            # מצא מדידות מצב רוח אחרי הפעילות (תוך 6 שעות)
            relevant_moods = mood_df[(mood_df["date"] >= act_date) & 
                                    (mood_df["date"] <= act_date + pd.Timedelta(hours=6))]
            
            if not relevant_moods.empty:
                # קח את ממוצע מצב הרוח אם ישנם מספר רשומות
                avg_mood = relevant_moods["severity"].mean()
                
                matched_data.append({
                    "date": act_date,
                    "activity_name": act_row["name"],
                    "duration": act_row["duration"],
                    "intensity_score": act_row["intensity_score"],
                    "activity_score": act_row["activity_score"],
                    "mood_after": avg_mood
                })
        
        if len(matched_data) < 3:
            return "Not enough matched activity-mood data for analysis."
        
        matched_df = pd.DataFrame(matched_data)
        
        # קבץ לפי שם פעילות
        activity_analysis = []
        
        for activity_name, group in matched_df.groupby("activity_name"):
            if len(group) >= 2:  # לפחות 2 מופעים
                # בדוק שם פעילות תקין
                if not activity_name or len(activity_name) < 2:
                    continue
                
                avg_duration = group["duration"].mean()
                avg_score = group["activity_score"].mean()
                avg_mood = group["mood_after"].mean()
                
                # מתאם בין ציון פעילות ומצב רוח (אם יש מספיק נקודות נתונים)
                correlation = None
                if len(group) >= 3:
                    if group["activity_score"].std() > 0 and group["mood_after"].std() > 0:
                        correlation, p_value = pearsonr(group["activity_score"], group["mood_after"])
                        # אם הקורלציה לא מובהקת (p-value גבוה), אל תציג אותה
                        if p_value > 0.2:
                            correlation = None
                
                activity_analysis.append({
                    "activity_name": activity_name,
                    "count": len(group),
                    "avg_duration": round(avg_duration, 1),
                    "avg_mood_after": round(avg_mood, 2),
                    "correlation": round(correlation, 3) if correlation is not None else None
                })
        
        # מיין לפי השפעת מצב רוח (הגבוה ביותר תחילה)
        activity_analysis.sort(key=lambda x: x["avg_mood_after"], reverse=True)
        
        return activity_analysis
    except Exception as e:
        return f"Error in activity pattern analysis: {str(e)}"

def analyze_medication_patterns(data, mood_field):
    if not data or "medications" not in data or "symptoms" not in data:
        return "Not enough data for medication pattern analysis."
    
    try:
        # חילוץ נתוני תרופות
        med_data = []
        for item in data.get("medications", []):
            if "date" in item and "name" in item:
                med_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "name": item["name"],
                    "quantity": item.get("quantity", 1)
                })
        
        # קבל נתוני מצב רוח/סימפטום
        mood_data = []
        for item in data["symptoms"]:
            if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item.get("severity", 0)
                })
        
        if len(med_data) < 5 or len(mood_data) < 5:
            return "Not enough data points for medication analysis."
        
        # צור DataFrames
        med_df = pd.DataFrame(med_data)
        mood_df = pd.DataFrame(mood_data)
        
        # קבץ לפי יום ליצירת טרנזקציות
        med_df["day"] = med_df["date"].dt.date
        mood_df["day"] = mood_df["date"].dt.date
        
        # צור סט נתוני טרנזקציות
        days = sorted(set(list(med_df["day"]) + list(mood_df["day"])))
        transactions = []
        
        for day in days:
            day_meds = med_df[med_df["day"] == day]["name"].unique().tolist()
            
            # עבור מצב רוח, קבל את הממוצע של אותו יום
            day_mood_df = mood_df[(mood_df["day"] == day) & (mood_df["type"] == mood_field)]
            
            if not day_mood_df.empty:
                avg_severity = day_mood_df["severity"].mean()
                mood_level = f"{mood_field}_Level_{round(avg_severity)}"
                transaction = day_meds + [mood_level]
                transactions.append(transaction)
        
        if len(transactions) < 5:
            return "Not enough daily data for pattern analysis."
        
        # הפעל אלגוריתם Apriori עבור סטים תדירים
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # מצא סטים תדירים
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        
        if frequent_itemsets.empty:
            return "No significant patterns found with current support threshold."
        
        # צור חוקי אסוציאציה
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        if rules.empty:
            return "No strong association rules found."
        
        # סנן חוקים הקשורים לרמות מצב רוח
        mood_rules = []
        for _, rule in rules.iterrows():
            antecedents = list(rule["antecedents"])
            # בדיקה אם אנטיצדנט או קונסקוונט מכילים דירוגי רמת מצב רוח
            mood_level_pattern = f"{mood_field}_Level_"
            has_mood = False
            
            for item in list(rule["antecedents"]) + list(rule["consequents"]):
                if isinstance(item, str) and item.startswith(mood_level_pattern):
                    has_mood = True
                    break
            
            if has_mood:
                rule_dict = {
                    "antecedents": list(rule["antecedents"]),
                    "consequents": list(rule["consequents"]),
                    "confidence": rule["confidence"],
                    "lift": rule["lift"]
                }
                mood_rules.append(rule_dict)
        
        if len(mood_rules) == 0:
            return "No significant medication-mood associations found."
            
        # מיון לפי lift (חשיבות)
        mood_rules.sort(key=lambda x: x["lift"], reverse=True)
        
        return mood_rules[:5]  # החזר 5 חוקים עליונים
    except Exception as e:
        return f"Error in medication pattern analysis: {str(e)}"

# פונקציות ניתוח עבור ממשק המשתמש
def activity_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
        
    # נתח את הפעילויות בצורה בסיסית (לפי הממשק הפשוט)
    activity_df, mood_df = prepare_activity_and_mood_data(translated_data_global, mood_field)
    basic_insights = generate_activity_insights(activity_df, mood_df)
    
    # אם אין מספיק נתונים, החזר את הניתוח הבסיסי
    if "Not enough data" in basic_insights or "Not enough close-timing data" in basic_insights:
        return basic_insights
    
    # נתח את הפעילויות בצורה מתקדמת
    advanced_analysis = analyze_activity_patterns(translated_data_global, mood_field)
    
    # אם יש שגיאה או אין מספיק נתונים, החזר רק את הניתוח הבסיסי
    if isinstance(advanced_analysis, str):
        if "Not enough" in advanced_analysis:
            return basic_insights
        return basic_insights + "\n\n" + advanced_analysis
    
    # אם אין פעילויות לניתוח, החזר רק את הניתוח הבסיסי
    if not advanced_analysis:
        return basic_insights
    
    # בנה תובנות מפורטות על סמך הניתוח המתקדם
    detailed_insights = "\n\nDetailed Activity Analysis:\n"
    for activity in advanced_analysis[:3]:
        # וודא שיש שם פעילות תקין
        activity_name = activity.get('activity_name', '')
        if not activity_name or len(activity_name) < 2 or activity_name == "Unknown":
            continue
            
        # וודא שיש לפחות 2 מופעים של הפעילות
        if activity.get('count', 0) < 2:
            continue
            
        detailed_insights += f"- {activity_name}: {activity.get('avg_mood_after', 0):.1f}/5 rating after {activity.get('count')} activities\n"
        
        # רק אם יש קורלציה משמעותית וגם לפחות 3 מופעים, הצג אותה
        if activity.get('correlation') is not None and abs(activity.get('correlation', 0)) > 0.3 and activity.get('count', 0) >= 3:
            corr = activity.get('correlation')
            direction = "positive" if corr > 0 else "negative"
            explanation = "higher intensity = better mood" if corr > 0 else "lower intensity = better mood"
            detailed_insights += f"  ({explanation}, correlation: {corr:.2f})\n"
    
    # רק אם נוספו תובנות מפורטות מעבר לכותרת, החזר אותן
    if detailed_insights != "\n\nDetailed Activity Analysis:\n":
        return basic_insights + detailed_insights
    else:
        return basic_insights

def medication_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    medication_df, mood_df = prepare_medication_and_mood_data(translated_data_global, mood_field)
    basic_insights = generate_medication_insights(medication_df, mood_df)
    
    # שלב את התובנות הבסיסיות עם הניתוח המתקדם
    advanced_analysis = analyze_medication_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        if "Not enough" in advanced_analysis or "No significant" in advanced_analysis:
            return basic_insights
        return basic_insights + "\n\n" + advanced_analysis
    
    detailed_insights = "\n\nDetailed Medication Patterns:\n"
    for idx, rule in enumerate(advanced_analysis[:3]):
        antecedents = list(rule.get("antecedents", []))
        consequents = list(rule.get("consequents", []))
        
        meds = []
        mood_level = None
        
        for item in antecedents + consequents:
            if isinstance(item, str):
                if item.startswith(f"{mood_field}_Level_"):
                    mood_level = item.replace(f"{mood_field}_Level_", "Rating: ")
                else:
                    meds.append(item)
        
        if meds and mood_level:
            meds_str = ", ".join(meds)
            detailed_insights += f"- {meds_str} associated with {mood_level}\n"
            detailed_insights += f"  (Confidence: {rule.get('confidence', 0):.2f}, Lift: {rule.get('lift', 0):.2f})\n"
    
    if detailed_insights != "\n\nDetailed Medication Patterns:\n":
        return basic_insights + detailed_insights
    else:
        return basic_insights

def symptom_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    symptom_df, mood_df = prepare_symptom_and_mood_data(translated_data_global, mood_field)
    return generate_symptom_insights(symptom_df, mood_df)

# פונקציות עיבוד קובץ
def upload_json(file_obj):
    global translated_data_global
    if file_obj is None:
        return None, "❌ No file uploaded."
    try:
        # נשתמש בפונקציה המקורית שכוללת תרגום וניתוח תזונתי
        processed_file, status = upload_and_process(file_obj)
        return processed_file, status
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# יצירת האפליקציה עם העיצוב החדש
with gr.Blocks(title="Parkinson's Health Pattern Analysis") as app:
    gr.Markdown("# 📈 Parkinson's Health Pattern Analysis")

    with gr.Row():
        file_input = gr.File(label="Upload JSON File")
        upload_button = gr.Button("Upload and Process", variant="primary", size="lg")
    
    output_text = gr.Textbox(label="Status", interactive=False)
    processed_file = gr.File(label="Download Processed File", interactive=False)

    mood_selector = gr.Dropdown(
        ["Parkinson's State", "Physical State", "My Mood"],
        label="Select Mood Field",
        value="My Mood"
    )

    with gr.Tabs():
        with gr.TabItem("🏃 Activity Analysis"):
            activity_button = gr.Button("Analyze Activity Patterns", variant="primary")
            activity_output = gr.Markdown(label="Activity Insights")
        
        with gr.TabItem("💊 Medication Analysis"):
            medication_button = gr.Button("Analyze Medication Patterns", variant="primary")
            medication_output = gr.Markdown(label="Medication Insights")

        with gr.TabItem("🩺 Symptom Analysis"):
            symptom_button = gr.Button("Analyze Symptom Patterns", variant="primary")
            symptom_output = gr.Markdown(label="Symptom Insights")

    # קישור הפונקציות לכפתורים
    upload_button.click(fn=upload_json, inputs=[file_input], outputs=[processed_file, output_text])
    activity_button.click(fn=activity_analysis_summary, inputs=[mood_selector], outputs=[activity_output])
    medication_button.click(fn=medication_analysis_summary, inputs=[mood_selector], outputs=[medication_output])
    symptom_button.click(fn=symptom_analysis_summary, inputs=[mood_selector], outputs=[symptom_output])

# הפעלת האפליקציה
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.launch(server_name='0.0.0.0', server_port=port)
