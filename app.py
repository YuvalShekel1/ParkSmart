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
    "דיסטוניה": "Dystonia",
    "דיסקינזיה": "Dyskinesia",
}

# פונקציה לבדיקת תקינות שמות (פעילויות/תרופות וכו')
def is_valid_name(name):
    """
    בדיקה קפדנית יותר של שמות. מחזירה True רק עבור שמות תקינים לחלוטין.
    """
    if not name or not isinstance(name, str) or len(name) < 2:
        return False
    
    # בדוק שיש לפחות 50% אותיות באנגלית או עברית
    english_letters = sum(1 for c in name if c.isalpha() and ord(c) < 128)
    hebrew_letters = sum(1 for c in name if '\u0590' <= c <= '\u05FF')
    total_chars = len(name)
    
    letter_percentage = (english_letters + hebrew_letters) / total_chars
    
    # חייב להיות לפחות 50% אותיות תקינות
    if letter_percentage < 0.5:
        return False
    
    # רשימה של שמות ידועים תקינים - תוסיף לפי הצורך
    known_valid_names = [
        "tennis", "walking", "swimming", "yoga", "running", 
        "strength training", "pilates", "cycling", "hiking",
        "טניס", "הליכה", "שחייה", "יוגה", "ריצה", 
        "אימון כוח", "פילאטיס", "רכיבה על אופניים", "טיול",
        "azilect", "dopicar", "sinemet", "rasagiline", "levodopa",
        "אזילקט", "דופיקר", "סינמט", "רסאג'ילין", "לבודופה",
        "tremor", "stiffness", "slowness", "balance problems",
        "רעד", "נוקשות", "איטיות", "בעיות שיווי משקל",
        "dystonia", "dyskinesia", "fatigue", "pain",
        "דיסטוניה", "דיסקינזיה", "עייפות", "כאב",
    ]
    
    # בדוק אם השם מכיל אחד השמות התקינים הידועים
    for valid_name in known_valid_names:
        if valid_name.lower() in name.lower():
            return True
    
    # תבנית נפוצה לשמות לא תקינים - אותיות מוזרות
    strange_chars = sum(1 for c in name if ord(c) > 255 and not '\u0590' <= c <= '\u05FF')
    if strange_chars > 0:
        return False
    
    # אם עברנו את כל הבדיקות והשם מכיל בעיקר אותיות אנגליות/עבריות, הוא כנראה תקין
    return True

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

# פונקציות יצירת תובנות בסיסיות - הגרסה המשופרת
def generate_activity_insights(activity_df, mood_df):
    insights = "🏃 Activity Insights:\n"

    if activity_df.empty or mood_df.empty:
        return insights + "• No activities data available.\n"

    # ניתוח פעילויות - התמקדות רק בפעילויות תקינות ופופולריות
    all_activities = activity_df["item"].apply(lambda x: x.get("activityName", "Unknown"))
    
    # רשימה של פעילויות תקינות (רק באנגלית ועברית)
    valid_activities = []
    activity_counts = {}
    
    for activity in all_activities:
        if is_valid_name(activity):
            valid_activities.append(activity)
            if activity in activity_counts:
                activity_counts[activity] += 1
            else:
                activity_counts[activity] = 1
    
    # מיון פעילויות לפי שכיחות
    sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    
    # התמקד רק בפעילויות המשמעותיות (עם לפחות 2 מופעים)
    significant_activities = [a for a, count in sorted_activities if count >= 2]
    
    # אם יש פעילויות משמעותיות, הצג רק אותן
    if significant_activities:
        insights += "• Main activities:\n"
        for activity in significant_activities[:3]:  # הצג רק את 3 הפעילויות העיקריות
            insights += f"  - {activity}\n"
    
    # ניתוח קשר בין פעילויות למצב רוח - רק עבור פעילויות תקינות
    combined_data = []
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        
        # קח פעילויות מאותו יום
        same_day_activities = activity_df[activity_df["date"].dt.date == mood_date.date()]
        
        for _, act_row in same_day_activities.iterrows():
            activity_item = act_row["item"]
            activity_name = activity_item.get("activityName", "Unknown")
            
            # בדיקה קפדנית יותר של תקינות השם
            if is_valid_name(activity_name):
                combined_data.append({
                    "mood_value": mood_value,
                    "activity_name": activity_name
                })

    if combined_data:
        analysis_df = pd.DataFrame(combined_data)
        
        # תובנות ספציפיות על השפעת פעילויות
        insights += "\n• Activity impact on your state:\n"
        
        activity_mood_impact = {}
        overall_mood_avg = mood_df["value"].mean() if not mood_df.empty else 0
        
        for activity in significant_activities:
            activity_data = analysis_df[analysis_df["activity_name"] == activity]
            if len(activity_data) >= 2:  # לפחות 2 מופעים
                avg_mood = activity_data["mood_value"].mean()
                diff = avg_mood - overall_mood_avg
                
                # נ
