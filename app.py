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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

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
    "אימון טנש": "Tens Training",
    "אימון טנש קבוצתי": "Group Tens Training",
    "משעה 2020 3 משחקים. הפסקה של 15 דקות לפני המשחקים": "From 8:20 PM, 3 games. 15-minute break before the games",
    "טנש": "Tens",
    "טאקי": "Taki (card game)",
    "טורניר טנש": "Tens tournament",
    "הרכבת כסאות גינה": "Assembling garden chairs",
    "נסיעה לבית שאן": "Trip to Beit Shean",
    "סיור במוזיאון גולני": "Tour at Golani Museum",
    "עבודת גינה": "Garden work",
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


def prepare_medication_and_mood_data(data, mood_field):
    """
    פונקציה מעודכנת להכנת הנתונים - בודקת גם את השדה 'medicines' וגם 'medications'
    """
    if not data:
        return pd.DataFrame(), pd.DataFrame()

    # בדיקה אם קיים medicines או medications
    medications_list = []
    
    # בדוק אם השדה 'medications' קיים
    if "medications" in data and isinstance(data["medications"], list):
        for item in data["medications"]:
            if "date" in item or "dateTaken" in item:
                # שימוש ב-dateTaken אם date לא קיים
                date_field = item.get("date", item.get("dateTaken"))
                if date_field:
                    medications_list.append({
                        "date": pd.to_datetime(date_field),
                        "item": item
                    })
    
    # בדוק גם את השדה 'medicines' אם ה'medications' ריק או אפילו אם לא ריק
    if "medicines" in data and isinstance(data["medicines"], list):
        for item in data["medicines"]:
            if "date" in item or "dateTaken" in item:
                # שימוש ב-dateTaken אם date לא קיים
                date_field = item.get("date", item.get("dateTaken"))
                if date_field:
                    medications_list.append({
                        "date": pd.to_datetime(date_field),
                        "item": item
                    })
    
    medication_df = pd.DataFrame(medications_list)

    # בדוק את כל הדיווחים ולא רק את אלה שמתאימים לשדה שהתבקש
    mood_list = []
    
    # חפש בשדה feelings
    if "feelings" in data and isinstance(data["feelings"], list):
        for item in data["feelings"]:
            if "date" in item and "severity" in item:
                # בדוק אם יש שדה type שמתאים לבקשה או אין שדה כזה
                if item.get("type") == mood_field:
                    mood_list.append({
                        "date": pd.to_datetime(item["date"]),
                        "value": item["severity"]
                    })

    # נסה ללא פילטור אם יש מעט תוצאות
    if len(mood_list) < 3 and "feelings" in data and isinstance(data["feelings"], list):
        mood_list = []
        for item in data["feelings"]:
            if "date" in item and "severity" in item:
                mood_list.append({
                    "date": pd.to_datetime(item["date"]),
                    "value": item["severity"]
                })
    
    mood_df = pd.DataFrame(mood_list)

    print(f"Found {len(medication_df)} medication records and {len(mood_df)} mood records")
    
    return medication_df, mood_df

def prepare_symptom_and_mood_data(data, mood_field):
    if not data or "symptoms" not in data or "feelings" not in data:
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
    for item in data.get("feelings", []):
        if "date" in item and item.get("type") == mood_field and "severity" in item:
            mood_list.append({
                "date": pd.to_datetime(item["date"]),
                "value": item["severity"]
            })
    mood_df = pd.DataFrame(mood_list)

    return symptom_df, mood_df



from sklearn.linear_model import LinearRegression

def generate_medication_insights(medication_df, mood_df):
    insights = "💊 Medication Insights:\n"

    if medication_df.empty or mood_df.empty:
        return insights + "• No medication data available.\n"

    # ניתוח תרופות - סינון שמות לא תקינים
    all_medications = medication_df["item"].apply(lambda x: x.get("name", "Unknown"))
    
    # סינון שמות תרופות לא תקינים
    valid_medications = []
    medication_counts = {}
    
    for medication in all_medications:
        # בדוק אם שם התרופה תקין
        if medication and isinstance(medication, str):
            is_valid = all(c.isalnum() or c.isspace() or '\u0590' <= c <= '\u05FF' or c in [',', '.', '-', '(', ')'] for c in medication)
            if is_valid and len(medication) >= 2:
                valid_medications.append(medication)
                if medication in medication_counts:
                    medication_counts[medication] += 1
                else:
                    medication_counts[medication] = 1
    
    # מיון תרופות לפי תדירות
    sorted_medications = sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_medications:
        insights += "• Medication frequency:\n"
        for medication, count in sorted_medications:
            if count > 0:
                insights += f"  - {medication}: {count} times\n"
    
    # אם יש נתוני מינון, נוסיף ניתוח מינון
    if medication_df["item"].apply(lambda x: "quantity" in x).any():
        insights += "\n• Medication dosages:\n"
        
        for medication, count in sorted_medications:
            if count > 0:
                med_items = medication_df[medication_df["item"].apply(lambda x: x.get("name", "") == medication)]
                quantities = med_items["item"].apply(lambda x: float(x.get("quantity", 0)))
                
                if not quantities.empty and quantities.sum() > 0:
                    min_dose = quantities.min()
                    max_dose = quantities.max()
                    avg_dose = quantities.mean()
                    
                    if min_dose == max_dose:
                        insights += f"  - {medication}: Consistent dosage of {min_dose}\n"
                    else:
                        insights += f"  - {medication}: Varies between {min_dose} and {max_dose} (avg: {round(avg_dose, 1)})\n"
    
    # ניתוח השפעת תרופות על מצב רוח
    combined_data = []
    
    # קח שילובים של מצב רוח ותרופות של אותו יום
    for _, mood_row in mood_df.iterrows():
        mood_date = mood_row["date"]
        mood_value = mood_row["value"]
        
        same_day_meds = medication_df[medication_df["date"].dt.date == mood_date.date()]
        
        if not same_day_meds.empty:
            for _, med_row in same_day_meds.iterrows():
                med_item = med_row["item"]
                med_name = med_item.get("name", "Unknown")
                
                # סינון שמות תרופות לא תקינים
                is_valid = False
                if med_name and isinstance(med_name, str):
                    is_valid = all(c.isalnum() or c.isspace() or '\u0590' <= c <= '\u05FF' or c in [',', '.', '-', '(', ')'] for c in med_name)
                
                if is_valid and len(med_name) >= 2:
                    dosage = float(med_item.get("quantity", 0))
                    combined_data.append({
                        "mood_value": mood_value,
                        "medication_name": med_name,
                        "dosage": dosage
                    })
    
    if combined_data:
        analysis_df = pd.DataFrame(combined_data)
        
        # ניתוח השפעת תרופות על מצב רוח
        insights += "\n• Medication impact on mood state:\n"
        
        medication_mood_impact = {}
        medication_types = analysis_df["medication_name"].unique()
        
        for medication in medication_types:
            medication_data = analysis_df[analysis_df["medication_name"] == medication]
            if len(medication_data) >= 1:
                avg_mood = medication_data["mood_value"].mean()
                avg_mood_rounded = round(avg_mood, 1)
                mood_description = ""
                
                # הגדרת תיאור מצב רוח
                if avg_mood >= 4.5:
                    mood_description = "excellent"
                elif avg_mood >= 4:
                    mood_description = "very good"
                elif avg_mood >= 3.5:
                    mood_description = "good"
                elif avg_mood >= 3:
                    mood_description = "moderate"
                elif avg_mood >= 2:
                    mood_description = "below average"
                else:
                    mood_description = "poor"
                
                medication_mood_impact[medication] = {
                    "count": len(medication_data),
                    "avg_mood": avg_mood_rounded,
                    "description": mood_description
                }
        
        # מיון לפי השפעה על מצב רוח (מהגבוה לנמוך)
        sorted_impacts = sorted(medication_mood_impact.items(), key=lambda x: x[1]["avg_mood"], reverse=True)
        
        for medication, impact in sorted_impacts:
            insights += f"  - After {medication} ({impact['count']} times): Mood is {impact['description']} ({impact['avg_mood']}/5)\n"
        
        # ניתוח השפעת מינון על מצב רוח (אם יש נתונים מספיקים)
        dosage_insights = ""
        
        for medication in medication_types:
            medication_data = analysis_df[analysis_df["medication_name"] == medication]
            
            # בדוק אם יש מספיק נתונים ואם יש שונות במינון
            if len(medication_data) >= 3 and medication_data["dosage"].std() > 0:
                # חלוקה למינונים
                low_dosage = medication_data[medication_data["dosage"] <= medication_data["dosage"].median()]
                high_dosage = medication_data[medication_data["dosage"] > medication_data["dosage"].median()]
                
                if len(low_dosage) >= 1 and len(high_dosage) >= 1:
                    low_mood = low_dosage["mood_value"].mean()
                    high_mood = high_dosage["mood_value"].mean()
                    
                    if abs(low_mood - high_mood) >= 0.5:  # רק אם יש הבדל משמעותי
                        better_dosage = "Higher" if high_mood > low_mood else "Lower"
                        dosage_insights += f"  - {medication}: {better_dosage} dosage associated with better mood state\n"
                        dosage_insights += f"    (Low dosage: {round(low_mood, 1)}/5, High dosage: {round(high_mood, 1)}/5)\n"
        
        if dosage_insights:
            insights += "\n• Impact of medication dosage on mood:\n" + dosage_insights
    
    return insights

def generate_symptom_insights(symptom_df, mood_df, mood_field):
    insights = "🩺 Symptom Insights:\n"

    if symptom_df.empty or mood_df.empty:
        return insights + "• No symptom data available.\n"

    # דלה את כל סוגי הסימפטומים
    symptom_fields = set()
    for _, row in symptom_df.iterrows():
        item = row["item"]
        for key in item.keys():
            if key not in ["date", "notes", "id", "Parkinson's State", "My Mood", "Physical State", "type", "severity", "createdAt", "updatedAt", "__v", "_id", "userId"]:
                symptom_fields.add(key)
    
    # הוסף גם את הסימפטומים שמופיעים בשדה type
    for _, row in symptom_df.iterrows():
        item = row["item"]
        if "type" in item and item["type"] not in ["Parkinson's State", "My Mood", "Physical State"]:
            symptom_fields.add(item["type"])
    
    # הסר כפילויות
    symptom_fields = list(set(symptom_fields))
    
    # אם אין שדות סימפטומים, בדוק אם יש סימפטומים אחרים כלשהם
    if not symptom_fields:
        symptom_types = set()
        for _, row in symptom_df.iterrows():
            item = row["item"]
            if "type" in item:
                symptom_types.add(item["type"])
        
        if symptom_types:
            insights += "• Symptom record types:\n"
            for symptom_type in symptom_types:
                insights += f"  - {symptom_type}\n"
        else:
            insights += "• No specific symptom data detected.\n"
        
        return insights

    # עבור כל סימפטום, בדוק את ההשפעה על מצב הרוח
    symptom_effects = []
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
        
        # חשב את ההשפעה רק אם יש מספיק נתונים
        if symptom_present_moods and symptom_absent_moods:
            present_avg = np.mean(symptom_present_moods)
            absent_avg = np.mean(symptom_absent_moods)
            diff = present_avg - absent_avg
            direction = "higher" if diff > 0 else "lower"
            
            symptom_effects.append({
                "symptom": symptom,
                "present_count": len(symptom_present_moods),
                "absent_count": len(symptom_absent_moods),
                "present_avg": round(present_avg, 1),
                "absent_avg": round(absent_avg, 1),
                "diff": round(abs(diff), 1),
                "direction": direction,
                "significant": abs(diff) >= 0.3
            })
    
    # מיין את ההשפעות לפי גודל ההבדל
    symptom_effects.sort(key=lambda x: x["diff"], reverse=True)
    
    if symptom_effects:
        insights += f"• Symptom impact on {mood_field}:\n"
        
        for effect in symptom_effects:
            if effect["significant"]:
                insights += f"  - {effect['symptom']} ({effect['present_count']} occurrences): Mood {effect['direction']} by {effect['diff']} points when present\n"
                insights += f"    (Average mood: {effect['present_avg']}/5 with symptom, {effect['absent_avg']}/5 without)\n"
            else:
                insights += f"  - {effect['symptom']} ({effect['present_count']} occurrences): No significant mood impact\n"
    
    # ניתוח של יחסי גומלין בין סימפטומים שונים
    if len(symptom_fields) >= 2:
        # בדוק אילו סימפטומים מופיעים יחד
        symptom_co_occurrence = {}
        
        for i, symptom1 in enumerate(symptom_fields):
            for symptom2 in symptom_fields[i+1:]:
                pair_key = f"{symptom1}_{symptom2}"
                symptom_co_occurrence[pair_key] = {"both": 0, "only1": 0, "only2": 0, "none": 0}
                
                for _, row in symptom_df.iterrows():
                    item = row["item"]
                    date = row["date"].date()
                    
                    has_symptom1 = (symptom1 in item and item[symptom1]) or ("type" in item and item["type"] == symptom1)
                    has_symptom2 = (symptom2 in item and item[symptom2]) or ("type" in item and item["type"] == symptom2)
                    
                    if has_symptom1 and has_symptom2:
                        symptom_co_occurrence[pair_key]["both"] += 1
                    elif has_symptom1:
                        symptom_co_occurrence[pair_key]["only1"] += 1
                    elif has_symptom2:
                        symptom_co_occurrence[pair_key]["only2"] += 1
                    else:
                        symptom_co_occurrence[pair_key]["none"] += 1
        
        # זיהוי זוגות משמעותיים
        significant_pairs = []
        
        for pair_key, counts in symptom_co_occurrence.items():
            symptom1, symptom2 = pair_key.split("_")
            
            # חישוב מדד לקשר בין הסימפטומים
            total = counts["both"] + counts["only1"] + counts["only2"] + counts["none"]
            if total == 0:
                continue
                
            expected_both = (counts["both"] + counts["only1"]) * (counts["both"] + counts["only2"]) / total
            if expected_both == 0:
                continue
                
            association_strength = counts["both"] / expected_both
            
            if counts["both"] >= 2 and association_strength >= 1.5:
                significant_pairs.append({
                    "symptom1": symptom1,
                    "symptom2": symptom2,
                    "both_count": counts["both"],
                    "association_strength": round(association_strength, 1)
                })
        
        # מיון לפי חוזק הקשר
        significant_pairs.sort(key=lambda x: x["association_strength"], reverse=True)
        
        if significant_pairs:
            insights += "\n• Symptom co-occurrence patterns:\n"
            
            for pair in significant_pairs[:3]:  # הצג רק את 3 הזוגות המשמעותיים ביותר
                insights += f"  - {pair['symptom1']} and {pair['symptom2']} tend to occur together ({pair['both_count']} times)\n"
    
    return insights

# פונקציות ניתוח מתקדמות
def analyze_activity_patterns(data, mood_field):
    if not data or "activities" not in data or "feelings" not in data:
        return "Not enough data for activity pattern analysis."

    try:
        activity_data = []
        for item in data.get("activities", []):
            if "date" in item and "activityName" in item and "duration" in item and "intensity" in item:
                name = item["activityName"]
                if not name or len(name) < 2:
                    continue
                activity_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "activity_name": name,
                    "duration": item["duration"],
                    "intensity": item["intensity"]
                })

        mood_data = []
        for item in data["feelings"]:
            if "date" in item and item.get("type") == mood_field and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item["severity"]
                })

        if len(activity_data) < 3 or len(mood_data) < 3:
            return "Not enough data points for activity analysis."

        activity_df = pd.DataFrame(activity_data)
        mood_df = pd.DataFrame(mood_data)

        matched_data = []
        for _, act in activity_df.iterrows():
            end_of_day = act["date"].replace(hour=23, minute=59, second=59)
            relevant_moods = mood_df[(mood_df["date"] >= act["date"]) & (mood_df["date"] <= end_of_day)]
            if not relevant_moods.empty:
                avg_mood = relevant_moods["severity"].mean()
                matched_data.append({
                    "activity_name": act["activity_name"],
                    "duration": act["duration"],
                    "intensity": act["intensity"],
                    "mood_after": avg_mood
                })

        if len(matched_data) < 3:
            return "Not enough matched activity-mood data for analysis."
            
        # ספירת מספר התצפיות לכל סוג פעילות
        activity_counts = {}
        for item in matched_data:
            act_name = item["activity_name"]
            activity_counts[act_name] = activity_counts.get(act_name, 0) + 1
            
        # סינון רק פעילויות עם לפחות 2 תצפיות
        filtered_data = [item for item in matched_data if activity_counts[item["activity_name"]] >= 2]
        
        if len(filtered_data) < 3:
            return "Not enough matched data after filtering (minimum 2 samples per activity type)."

        df = pd.DataFrame(filtered_data)
        X = df[["activity_name", "duration", "intensity"]]
        y = df["mood_after"]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["activity_name", "intensity"])
        ], remainder='passthrough')

        model = make_pipeline(preprocessor, LinearRegression())
        model.fit(X, y)

        coefs = model.named_steps["linearregression"].coef_
        feature_names = model.named_steps["columntransformer"].get_feature_names_out()

        result = []
        for i, (name, coef) in enumerate(zip(feature_names, coefs)):
            feature_type = ""
            feature_value = ""
            
            if "activity_name" in name:
                feature_type = "activity_name"
                feature_value = name.split("_")[-1]  # קח רק את השם האחרון אחרי ה-_
            elif "intensity" in name:
                feature_type = "intensity"
                feature_value = name.split("_")[-1]  # קח רק את השם האחרון אחרי ה-_
            else:
                feature_type = "duration"
                feature_value = ""
                
            result.append({
                "feature_type": feature_type,
                "feature_value": feature_value,
                "effect": round(coef, 2)
            })

        # ===== ניתוח מפורט נוסף: השפעות של משך זמן ספציפי =====
        try:
            # יצירת קטגוריות משך זמן
            df["duration_category"] = pd.cut(
                df["duration"].astype(float),
                bins=[0, 30, 60, float('inf')],
                labels=["short", "medium", "long"]
            )
            
            # מיפוי תיאורים
            duration_labels = {
                "short": "less than 30 minutes",
                "medium": "between 30-60 minutes",
                "long": "more than 60 minutes"
            }
            
            # ניתוח לכל סוג פעילות, עם חלוקה למשך זמן
            activity_duration_insights = []
            
            for activity in df["activity_name"].unique():
                activity_data = df[df["activity_name"] == activity]
                
                # המשך רק אם יש מספיק נתונים
                if len(activity_data) >= 3:
                    overall_avg = activity_data["mood_after"].mean()
                    
                    # בדיקת השפעות משך זמן שונה
                    for duration_cat in ["short", "medium", "long"]:
                        duration_data = activity_data[activity_data["duration_category"] == duration_cat]
                        
                        # בדוק שיש לפחות 2 מופעים (דפוס חוזר)
                        if len(duration_data) >= 2:
                            avg_mood = duration_data["mood_after"].mean()
                            effect = avg_mood - overall_avg
                            
                            # בדוק אם ההשפעה משמעותית
                            if abs(effect) >= 0.1:
                                activity_duration_insights.append({
                                    "activity": activity,
                                    "duration_desc": duration_labels[duration_cat],
                                    "effect": effect,
                                    "avg_mood": avg_mood,
                                    "count": len(duration_data)
                                })
            
            # ניתוח השפעות של עצימות ספציפית
            activity_intensity_insights = []
            
            for activity in df["activity_name"].unique():
                activity_data = df[df["activity_name"] == activity]
                
                # המשך רק אם יש מספיק נתונים
                if len(activity_data) >= 3:
                    overall_avg = activity_data["mood_after"].mean()
                    
                    # בדיקת השפעות עצימות שונה
                    for intensity in activity_data["intensity"].unique():
                        intensity_data = activity_data[activity_data["intensity"] == intensity]
                        
                        # בדוק שיש לפחות 2 מופעים (דפוס חוזר)
                        if len(intensity_data) >= 2:
                            avg_mood = intensity_data["mood_after"].mean()
                            effect = avg_mood - overall_avg
                            
                            # בדוק אם ההשפעה משמעותית
                            if abs(effect) >= 0.1:
                                activity_intensity_insights.append({
                                    "activity": activity,
                                    "intensity": intensity,
                                    "effect": effect,
                                    "avg_mood": avg_mood,
                                    "count": len(intensity_data)
                                })
            
            # שילוב של משך זמן ועצימות
            activity_complex_insights = []
            
            for activity in df["activity_name"].unique():
                activity_data = df[df["activity_name"] == activity]
                
                # המשך רק אם יש מספיק נתונים
                if len(activity_data) >= 4:
                    overall_avg = activity_data["mood_after"].mean()
                    
                    # בדיקת שילובים של משך ועצימות
                    for duration_cat in ["short", "medium", "long"]:
                        for intensity in activity_data["intensity"].unique():
                            combo_data = activity_data[
                                (activity_data["duration_category"] == duration_cat) & 
                                (activity_data["intensity"] == intensity)
                            ]
                            
                            # בדוק שיש לפחות 2 מופעים (דפוס חוזר)
                            if len(combo_data) >= 2:
                                avg_mood = combo_data["mood_after"].mean()
                                effect = avg_mood - overall_avg
                                
                                # בדוק אם ההשפעה משמעותית
                                if abs(effect) >= 0.1:
                                    activity_complex_insights.append({
                                        "activity": activity,
                                        "duration_desc": duration_labels[duration_cat],
                                        "intensity": intensity,
                                        "effect": effect,
                                        "avg_mood": avg_mood,
                                        "count": len(combo_data)
                                    })
            
            # הוספת התובנות המפורטות לתוצאה
            for insight in activity_duration_insights:
                effect = insight["effect"]
                if abs(effect) >= 0.2:  # מציג רק השפעות משמעותיות
                    direction = "increases" if effect > 0 else "decreases"
                    effect_size = abs(round(effect, 1))
                    
                    result.append({
                        "feature_type": "detailed_duration",
                        "feature_value": f"{insight['activity']} {insight['duration_desc']}",
                        "effect": effect if effect > 0 else -effect_size  # שומר על פורמט עקבי
                    })
            
            for insight in activity_intensity_insights:
                effect = insight["effect"]
                if abs(effect) >= 0.2:  # מציג רק השפעות משמעותיות
                    direction = "increases" if effect > 0 else "decreases"
                    effect_size = abs(round(effect, 1))
                    
                    result.append({
                        "feature_type": "detailed_intensity",
                        "feature_value": f"{insight['activity']} with {insight['intensity']} intensity",
                        "effect": effect if effect > 0 else -effect_size  # שומר על פורמט עקבי
                    })
            
            for insight in activity_complex_insights:
                effect = insight["effect"]
                if abs(effect) >= 0.2:  # מציג רק השפעות משמעותיות
                    direction = "increases" if effect > 0 else "decreases"
                    effect_size = abs(round(effect, 1))
                    
                    result.append({
                        "feature_type": "detailed_combo",
                        "feature_value": f"{insight['activity']} {insight['duration_desc']} with {insight['intensity']} intensity",
                        "effect": effect if effect > 0 else -effect_size  # שומר על פורמט עקבי
                    })
        
        except Exception as e:
            # במקרה של שגיאה, המשך עם התוצאות הקיימות
            pass

        # מיון התוצאות לפי גודל ההשפעה (מוחלט)
        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)

        return result
    except Exception as e:
        return f"Error in activity pattern analysis: {str(e)}"
def analyze_medication_patterns(data, mood_field):
    """
    ניתוח דפוסי תרופות והשפעתן על מצב הרוח/פרקינסון באמצעות רגרסיה לינארית
    עם דרישה מופחתת של מינימום 2 תצפיות במקום 3
    """
    if not data:
        return "No data available for analysis."
        
    # בדוק אם יש נתוני תרופות בשדה medicines או medications
    medications_data = None
    if "medications" in data and isinstance(data["medications"], list) and len(data["medications"]) > 0:
        medications_data = data["medications"]
    elif "medicines" in data and isinstance(data["medicines"], list) and len(data["medicines"]) > 0:
        medications_data = data["medicines"]
    
    if not medications_data:
        return "No medication data found. Please check your data structure."
    
    # בדוק אם יש נתוני מצב רוח
    feelings_data = None
    if "feelings" in data and isinstance(data["feelings"], list) and len(data["feelings"]) > 0:
        feelings_data = data["feelings"]
    
    if not feelings_data:
        return "No mood data found. Please check your data structure."
    
    try:
        # חילוץ נתוני תרופות
        medication_data = []
        for item in medications_data:
            # שימוש בשדה date או dateTaken
            date_field = item.get("date", item.get("dateTaken"))
            if not date_field:
                continue
                
            med_name = item.get("name", "")
            
            # סינון שמות תרופות לא תקינים - אבל מקל יותר
            if not med_name or len(med_name) < 2:
                continue
            
            # קטגוריזציה של מינונים
            categorized_name = med_name
            quantity = float(item.get("quantity", 0))
            
            # קטגוריזציה של דופיקר לפי מינון
            if "דופיקר" in med_name:
                if quantity >= 250:
                    categorized_name = "דופיקר מינון גבוה"
                elif quantity >= 175:
                    categorized_name = "דופיקר מינון בינוני"
                else:
                    categorized_name = "דופיקר מינון נמוך"
            
            medication_data.append({
                "date": pd.to_datetime(date_field),
                "medication_name": categorized_name,
                "original_name": med_name,
                "quantity": quantity
            })

        # חילוץ נתוני מצב רוח (כל סוגי הדיווחים)
        mood_data = []
        for item in feelings_data:
            if "date" in item and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item["severity"],
                    "type": item.get("type", "Unknown")
                })

        # סינון לפי סוג המצב רק אם יש מספיק נתונים
        if any(item["type"] == mood_field for item in mood_data):
            mood_data = [item for item in mood_data if item["type"] == mood_field]
        
        print(f"Found {len(medication_data)} medication records and {len(mood_data)} mood records")
        
        # שינוי כאן: מינימום 2 במקום 3
        if len(medication_data) < 2 or len(mood_data) < 2:
            return f"Not enough data points for medication analysis. Found {len(medication_data)} medication records and {len(mood_data)} mood records."

        medication_df = pd.DataFrame(medication_data)
        mood_df = pd.DataFrame(mood_data)

        # התאמת תרופות למצב רוח באותו יום או בטווח זמן סביר
        matched_data = []
        
        for _, med in medication_df.iterrows():
            med_date = med["date"]
            
            # חפש דיווחי מצב רוח עד 3 שעות אחרי התרופה
            relevant_moods = mood_df[(mood_df["date"] >= med_date) & 
                                    (mood_df["date"] <= med_date + pd.Timedelta(hours=3))]
            
            # אם אין קרובים, חפש באותו יום
            if relevant_moods.empty:
                end_of_day = med_date.replace(hour=23, minute=59, second=59)
                relevant_moods = mood_df[(mood_df["date"] >= med_date) & 
                                        (mood_df["date"] <= end_of_day)]
            
            if not relevant_moods.empty:
                avg_mood = relevant_moods["severity"].mean()
                
                # מצא את פרק הזמן בין נטילת התרופה לדיווח הראשון
                first_mood = relevant_moods.iloc[0]
                time_diff_hours = (first_mood["date"] - med_date).total_seconds() / 3600
                
                matched_data.append({
                    "medication_name": med["medication_name"],
                    "original_name": med["original_name"],
                    "quantity": med["quantity"],
                    "mood_after": avg_mood,
                    "time_diff_hours": time_diff_hours
                })

        # שינוי כאן: מינימום 2 במקום 3
        if len(matched_data) < 2:
            return f"Not enough matched medication-mood data for analysis. Found only {len(matched_data)} matches."
            
        # ספירת מספר התצפיות לכל סוג תרופה
        medication_counts = {}
        for item in matched_data:
            med_name = item["medication_name"]
            medication_counts[med_name] = medication_counts.get(med_name, 0) + 1
            
        # סינון רק תרופות עם לפחות 2 תצפיות (לא שינינו את זה כי זה כבר 2)
        filtered_data = [item for item in matched_data if medication_counts[item["medication_name"]] >= 2]
        
        # שינוי כאן: מינימום 2 במקום 3
        if len(filtered_data) < 2:
            return f"Not enough data after filtering for medications with at least 2 occurrences. Found only {len(filtered_data)} records."

        df = pd.DataFrame(filtered_data)
        
        # הכנת הנתונים לרגרסיה לינארית
        X = pd.get_dummies(df[["medication_name"]], drop_first=False)
        y = df["mood_after"]

        # רגרסיה לינארית
        model = LinearRegression()
        model.fit(X, y)

        # חילוץ המקדמים וחישוב ההשפעות
        result = []
        for i, (name, coef) in enumerate(zip(X.columns, model.coef_)):
            feature_type = "medication_name"
            
            # מקבלים שם קריא
            if "_" in name:
                feature_value = name.split("_", 1)[1]
            else:
                feature_value = name
                
            result.append({
                "feature_type": feature_type,
                "feature_value": feature_value,
                "effect": round(coef, 2)
            })

        # ניתוח השפעת מרווחי זמן - מקל בדרישות
        if "time_diff_hours" in df.columns and len(df) >= 4:
            try:
                # יוצרים קטגוריות של חלונות זמן
                df["time_window"] = pd.cut(
                    df["time_diff_hours"],
                    bins=[0, 1, 2, 4, float('inf')],
                    labels=["0-1 hour", "1-2 hours", "2-4 hours", "4+ hours"]
                )
                
                # נבדוק את ההשפעה של כל תרופה בחלונות זמן שונים
                for med in df["medication_name"].unique():
                    med_data = df[df["medication_name"] == med]
                    
                    # שינוי כאן: מינימום 2 במקום 3
                    if len(med_data) >= 2:
                        overall_avg = med_data["mood_after"].mean()
                        
                        # בדיקת השפעות חלונות זמן שונים
                        for window in ["0-1 hour", "1-2 hours", "2-4 hours", "4+ hours"]:
                            window_data = med_data[med_data["time_window"] == window]
                            
                            # בדוק שיש לפחות 1 מופע (מקל אף יותר)
                            if len(window_data) >= 1:
                                avg_mood = window_data["mood_after"].mean()
                                effect = avg_mood - overall_avg
                                
                                # בדוק אם ההשפעה משמעותית (מקל גם כאן)
                                if abs(effect) >= 0.1:
                                    result.append({
                                        "feature_type": "time_window",
                                        "feature_value": f"{med} within {window}",
                                        "effect": round(effect, 2)
                                    })
            except Exception as e:
                print(f"Error in time window analysis: {str(e)}")
                pass

        # ניתוח רצפי תרופות - מקל בדרישות גם כאן
        try:
            # ארגון התרופות לפי תאריך
            meds_by_date = {}
            for i, row in medication_df.iterrows():
                date_str = row["date"].date().isoformat()
                if date_str not in meds_by_date:
                    meds_by_date[date_str] = []
                meds_by_date[date_str].append({
                    "name": row["medication_name"],
                    "time": row["date"],
                    "index": i
                })
            
            # מציאת ימים עם יותר מתרופה אחת
            sequence_data = []
            for date, meds in meds_by_date.items():
                if len(meds) >= 2:
                    # מיין לפי זמן
                    sorted_meds = sorted(meds, key=lambda x: x["time"])
                    
                    # בדוק רצפים של שתי תרופות
                    for i in range(len(sorted_meds) - 1):
                        first = sorted_meds[i]["name"]
                        second = sorted_meds[i + 1]["name"]
                        sequence = f"{first} → {second}"
                        
                        # מצא דיווחים על מצב רוח לאחר הרצף
                        second_time = sorted_meds[i + 1]["time"]
                        moods_after = mood_df[mood_df["date"] > second_time]
                        same_day_end = pd.Timestamp(date + " 23:59:59")
                        same_day_moods = moods_after[moods_after["date"] <= same_day_end]
                        
                        if not same_day_moods.empty:
                            avg_mood = same_day_moods["severity"].mean()
                            sequence_data.append({
                                "sequence": sequence,
                                "mood": avg_mood,
                                "date": date
                            })
            
            # בדוק אם יש רצפים שמופיעים לפחות פעם אחת (מקל מאוד)
            if sequence_data:
                seq_df = pd.DataFrame(sequence_data)
                seq_counts = seq_df["sequence"].value_counts()
                common_sequences = seq_counts[seq_counts >= 1].index.tolist()
                
                for seq in common_sequences:
                    seq_mood_avg = seq_df[seq_df["sequence"] == seq]["mood"].mean()
                    # נשווה לממוצע הכללי
                    general_avg = seq_df["mood"].mean()
                    effect = seq_mood_avg - general_avg
                    
                    # מקל גם בהשפעה המינימלית
                    if abs(effect) >= 0.1:
                        result.append({
                            "feature_type": "medication_sequence",
                            "feature_value": seq,
                            "effect": round(effect, 2)
                        })
        except Exception as e:
            print(f"Error in medication sequence analysis: {str(e)}")
            pass

        # מיון התוצאות לפי גודל ההשפעה (מוחלט)
        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)

        return result
    except Exception as e:
        return f"Error in medication pattern analysis: {str(e)}"# פונקציות ניתוח עבור ממשק המשתמש
def activity_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."

    advanced_analysis = analyze_activity_patterns(translated_data_global, mood_field)

    if isinstance(advanced_analysis, str):
        return advanced_analysis

    if not advanced_analysis:
        return "No patterns found."

    mood_field_lower = mood_field.lower()
    header = f"## 🏃 **Activity impact on {mood_field}**\n\n"

    # מיון התובנות לפי סוג וכיוון השפעה
    green_insights = []
    red_insights = []
    neutral_insights = []
    
    # תובנות דפוסים מפורטים
    green_detailed_insights = []
    red_detailed_insights = []

    for item in advanced_analysis:
        feature_type = item.get("feature_type", "")
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect):.1f}"  # עיגול לספרה אחת אחרי הנקודה

        # קביעת הכותרת/תווית להצגה
        if feature_type == "activity_name":
            label = feature_value.strip().title()
        elif feature_type == "intensity":
            label = feature_value.strip().capitalize() + " intensity activity"
        elif feature_type == "duration":
            label = "Duration activity"
        elif feature_type == "detailed_duration":
            label = feature_value
        elif feature_type == "detailed_intensity":
            label = feature_value
        elif feature_type == "detailed_combo":
            label = feature_value
        else:
            label = feature_value

        # קביעת כיוון ותו
        if abs(effect) < 0.05:
            line = f"⚫ **{label}**: no significant impact\n\n"
            neutral_insights.append(line)
        elif effect > 0:
            if feature_type in ["detailed_duration", "detailed_intensity", "detailed_combo"]:
                line = f"🟢 **{label}** increases {mood_field_lower} by {effect_str} on average\n\n"
                green_detailed_insights.append(line)
            else:
                line = f"🟢 **{label}**: increases {mood_field_lower} by {effect_str} on average\n\n"
                green_insights.append(line)
        else:
            if feature_type in ["detailed_duration", "detailed_intensity", "detailed_combo"]:
                line = f"🔴 **{label}** decreases {mood_field_lower} by {effect_str} on average\n\n"
                red_detailed_insights.append(line)
            else:
                line = f"🔴 **{label}**: decreases {mood_field_lower} by {effect_str} on average\n\n"
                red_insights.append(line)

    # שילוב לפי סדר עדיפות
    basic_insights = header + "".join(green_insights + red_insights + neutral_insights)
    
    # בדוק אם יש תובנות מפורטות
    detailed_insights = ""
    if green_detailed_insights or red_detailed_insights:
        detailed_insights = "\n## Detailed Activity Patterns\n\n" + "".join(green_detailed_insights + red_detailed_insights)
    
    # שלב הכל ביחד
    combined_insights = basic_insights + detailed_insights
    
    return combined_insights

def medication_analysis_summary(mood_field):
    """
    מציג סיכום של ניתוח התרופות עם צבעים, בדומה לניתוח הפעילויות
    """
    if not translated_data_global:
        return "Please upload and process data first."
    
    # השתמש בפונקציה המקורית לקבלת תובנות בסיסיות
    medication_df, mood_df = prepare_medication_and_mood_data(translated_data_global, mood_field)
    basic_insights = generate_medication_insights(medication_df, mood_df)
    
    # ניתוח מתקדם של דפוסים בתרופות - דומה לפעילויות
    advanced_analysis = analyze_medication_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        return basic_insights + "\n\n" + advanced_analysis
    
    if not advanced_analysis:
        return basic_insights + "\n\nNo medication patterns found."
    
    # עיבוד התובנות בדיוק כמו בפעילויות
    mood_field_lower = mood_field.lower()
    header = f"\n## 💊 **Medication impact on {mood_field}**\n\n"
    
    green_insights = []
    red_insights = []
    neutral_insights = []
    
    # תובנות דפוסים מפורטים של חלונות זמן ורצפי תרופות
    green_detailed_insights = []
    red_detailed_insights = []
    
    for item in advanced_analysis:
        feature_type = item.get("feature_type", "")
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect):.1f}"  # עיגול לספרה אחת אחרי הנקודה
        
        # קביעת הכותרת/תווית להצגה
        if feature_type == "medication_name":
            label = feature_value.strip()
        elif feature_type == "time_window":
            label = feature_value
        elif feature_type == "medication_sequence":
            label = feature_value
        else:
            label = feature_value
        
        # קביעת כיוון ותו
        if abs(effect) < 0.05:
            line = f"⚫ **{label}**: no significant impact\n\n"
            neutral_insights.append(line)
        elif effect > 0:
            if feature_type in ["time_window", "medication_sequence"]:
                line = f"🟢 **{label}** increases {mood_field_lower} by {effect_str} on average\n\n"
                green_detailed_insights.append(line)
            else:
                line = f"🟢 **{label}**: increases {mood_field_lower} by {effect_str} on average\n\n"
                green_insights.append(line)
        else:
            if feature_type in ["time_window", "medication_sequence"]:
                line = f"🔴 **{label}** decreases {mood_field_lower} by {effect_str} on average\n\n"
                red_detailed_insights.append(line)
            else:
                line = f"🔴 **{label}**: decreases {mood_field_lower} by {effect_str} on average\n\n"
                red_insights.append(line)
    
    # שילוב לפי סדר עדיפות
    pattern_insights = header + "".join(green_insights + red_insights + neutral_insights)
    
    # בדוק אם יש תובנות מפורטות
    detailed_insights = ""
    if green_detailed_insights or red_detailed_insights:
        detailed_insights = "\n## Detailed Medication Patterns\n\n" + "".join(green_detailed_insights + red_detailed_insights)
    
    # שלב הכל ביחד
    combined_insights = basic_insights + pattern_insights + detailed_insights
    
    return combined_insights
    
def nutrition_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    
    nutrition_df = pd.DataFrame([
        {"date": pd.to_datetime(item["date"]), "item": item}
        for item in translated_data_global.get("nutritions", [])
        if "date" in item
    ])

    mood_df = pd.DataFrame([
        {"date": pd.to_datetime(item["date"]), "value": item["severity"]}
        for item in translated_data_global.get("symptoms", [])
        if item.get("type") == mood_field and "date" in item and "severity" in item
    ])

    if nutrition_df.empty or mood_df.empty:
        return "No data available for analysis."

    insights = "🍽️ Nutrition Insights:\n"
    combined = []

    for _, food_row in nutrition_df.iterrows():
        food_time = food_row["date"]
        food_name = food_row["item"].get("foodName", "Unknown")

        same_day_moods = mood_df[
            (mood_df["date"] >= food_time) & (mood_df["date"].dt.date == food_time.date())
        ]
        if not same_day_moods.empty:
            avg_mood = same_day_moods["value"].mean()
            combined.append((food_name, avg_mood))

    if not combined:
        return insights + "No mood data found after meals."

    df = pd.DataFrame(combined, columns=["food", "mood"])
    grouped = df.groupby("food").agg(["count", "mean"])
    grouped.columns = ["count", "avg_mood"]
    grouped = grouped[grouped["count"] >= 2].sort_values("avg_mood", ascending=False)

    for food, row in grouped.iterrows():
        mood_level = round(row["avg_mood"], 1)
        insights += f"- After eating {food} ({int(row['count'])} times): average {mood_field} = {mood_level}/5\n"
    # תובנות לפי רכיבים תזונתיים — בסגנון של "Mood higher by X points when present"
    insights += f"\n• Nutrient impact on {mood_field}:\n"

    nutrients = {
        "proteins": "Protein",
        "carbohydrates": "Carbohydrates",
        "fats": "Fats",
        "dietaryFiber": "Fiber"
    }

    thresholds = {
        "proteins": 10,
        "carbohydrates": 20,
        "fats": 10,
        "dietaryFiber": 3
    }

    enriched_data = []
    for _, food_row in nutrition_df.iterrows():
        food_time = food_row["date"]
        food_item = food_row["item"]
        food_name = food_item.get("foodName", "Unknown")
        nutrition = food_item.get("nutritionalValues", {})

        same_day_moods = mood_df[
            (mood_df["date"] >= food_time) & (mood_df["date"].dt.date == food_time.date())
        ]

        if same_day_moods.empty:
            continue

        avg_mood = same_day_moods["value"].mean()

        enriched_data.append({
            "food": food_name,
            "mood": avg_mood,
            **nutrition
        })

    df = pd.DataFrame(enriched_data)
    for key, label in nutrients.items():
        if key not in df.columns:
            continue

        threshold = thresholds[key]
        with_nutrient = df[df[key] >= threshold]
        without_nutrient = df[df[key] < threshold]

        if len(with_nutrient) < 2 or len(without_nutrient) < 2:
            # אם אין מספיק השוואה — דלג
            continue

        with_avg = with_nutrient["mood"].mean()
        without_avg = without_nutrient["mood"].mean()
        diff = round(with_avg - without_avg, 2)

        # אם אין הבדל מובהק — לא להציג
        if abs(diff) < 0.1:
            continue

        direction = "higher" if diff > 0 else "lower"
        insights += f"- {label} ({len(with_nutrient)} occurrences): {mood_field} {direction} by {abs(diff)} points when present\n"
        insights += f"  (Average {mood_field}: {round(with_avg, 1)}/5 with, {round(without_avg, 1)}/5 without)\n" 
       


    return insights


def symptom_analysis_summary(mood_field):
    if not translated_data_global:
        return "Please upload and process data first."
    symptom_df, mood_df = prepare_symptom_and_mood_data(translated_data_global, mood_field)
    return generate_symptom_insights(symptom_df, mood_df, mood_field)

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
            
        with gr.TabItem("🍽️ Nutrition Analysis"):
            nutrition_button = gr.Button("Analyze Nutrition Patterns", variant="primary")
            nutrition_output = gr.Markdown(label="Nutrition Insights")    

    # קישור הפונקציות לכפתורים
    upload_button.click(fn=upload_json, inputs=[file_input], outputs=[processed_file, output_text])
    activity_button.click(fn=activity_analysis_summary, inputs=[mood_selector], outputs=[activity_output])
    medication_button.click(fn=medication_analysis_summary, inputs=[mood_selector], outputs=[medication_output])
    symptom_button.click(fn=symptom_analysis_summary, inputs=[mood_selector], outputs=[symptom_output])
    nutrition_button.click(fn=nutrition_analysis_summary, inputs=[mood_selector], outputs=[nutrition_output])

# הפעלת האפליקציה
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.launch(server_name='0.0.0.0', server_port=port)
