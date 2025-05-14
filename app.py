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
def analyze_symptom_patterns(data, mood_field):
    """
    ניתוח דפוסי סימפטומים והשפעתם על מצב הרוח באמצעות רגרסיה לינארית
    """
    if not data:
        return "No data available for analysis."
        
    # בדוק אם יש נתוני סימפטומים
    symptoms_data = None
    if "symptoms" in data and isinstance(data["symptoms"], list) and len(data["symptoms"]) > 0:
        symptoms_data = data["symptoms"]
    
    if not symptoms_data:
        return "No symptom data found. Please check your data structure."
    
    # בדוק אם יש נתוני מצב רוח
    feelings_data = None
    if "feelings" in data and isinstance(data["feelings"], list) and len(data["feelings"]) > 0:
        feelings_data = data["feelings"]
    
    if not feelings_data:
        return "No mood data found. Please check your data structure."
    
    try:
        # המרת הנתונים לפורמט מתאים לניתוח
        symptom_data = []
        for item in symptoms_data:
            date_field = item.get("date", item.get("dateTaken"))
            if not date_field:
                continue
                
            # לוקחים רק סימפטומים ולא תחושות כלליות ולא "תסמינים אחרים"
            if "type" not in item or item["type"] in ["Parkinson's State", "My Mood", "Physical State", "Other Symptoms"]:
                continue
                
            severity = float(item.get("severity", 0))
            symptom_type = item.get("type", "Unknown")
            
            symptom_data.append({
                "date": pd.to_datetime(date_field),
                "symptom_type": symptom_type,
                "severity": severity
            })

        # חילוץ נתוני מצב רוח
        mood_data = []
        for item in feelings_data:
            if "date" in item and "severity" in item and item.get("type") == mood_field:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item["severity"]
                })
        
        print(f"Found {len(symptom_data)} symptom records and {len(mood_data)} mood records")
        
        # בדיקה שיש מספיק נתונים לניתוח
        if len(symptom_data) < 3 or len(mood_data) < 3:
            return f"Not enough data points for symptom analysis. Found {len(symptom_data)} symptom records and {len(mood_data)} mood records."

        symptom_df = pd.DataFrame(symptom_data)
        mood_df = pd.DataFrame(mood_data)

        # במקום לעשות רגרסיה לינארית מלאה, נעשה ניתוח פשוט יותר על כל סימפטום בנפרד
        result = []
        
        # נמצא את כל סוגי הסימפטומים הייחודיים
        symptom_types = symptom_df["symptom_type"].unique()
        
        for symptom_type in symptom_types:
            # נמצא את כל התאריכים עם הסימפטום הזה
            symptom_dates = symptom_df[symptom_df["symptom_type"] == symptom_type]["date"].dt.date.unique()
            
            # נמצא את מצב הרוח הממוצע בימים עם הסימפטום
            symptom_day_moods = []
            for date in symptom_dates:
                day_moods = mood_df[mood_df["date"].dt.date == date]["severity"]
                if not day_moods.empty:
                    symptom_day_moods.append(day_moods.mean())
            
            # נחשב את הממוצע של מצב הרוח בימים עם הסימפטום
            if len(symptom_day_moods) >= 2:  # לפחות 2 תצפיות
                symptom_avg_mood = sum(symptom_day_moods) / len(symptom_day_moods)
                
                # נחשב את הממוצע של מצב הרוח בימים ללא הסימפטום
                non_symptom_dates = [d for d in mood_df["date"].dt.date.unique() if d not in symptom_dates]
                non_symptom_day_moods = []
                for date in non_symptom_dates:
                    day_moods = mood_df[mood_df["date"].dt.date == date]["severity"]
                    if not day_moods.empty:
                        non_symptom_day_moods.append(day_moods.mean())
                
                # חישוב ההשפעה רק אם יש מספיק נתונים לימים ללא הסימפטום
                if len(non_symptom_day_moods) >= 2:
                    non_symptom_avg_mood = sum(non_symptom_day_moods) / len(non_symptom_day_moods)
                    
                    # ההשפעה היא ההפרש בין מצב הרוח הממוצע עם וללא הסימפטום
                    effect = symptom_avg_mood - non_symptom_avg_mood
                    
                    result.append({
                        "feature_type": "symptom_type",
                        "feature_value": symptom_type,
                        "effect": round(effect, 2),
                        "count": len(symptom_day_moods)
                    })
        
        # מיון התוצאות לפי גודל ההשפעה (מוחלט)
        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)

        return result
    except Exception as e:
        return f"Error in symptom pattern analysis: {str(e)}"

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
    
    # השתמש בפונקציה המקורית לקבלת תובנות בסיסיות - אבל לא משתמש בהן בתוצאה הסופית
    medication_df, mood_df = prepare_medication_and_mood_data(translated_data_global, mood_field)
    
    # ניתוח מתקדם של דפוסים בתרופות
    advanced_analysis = analyze_medication_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        return advanced_analysis
    
    if not advanced_analysis:
        return "No medication patterns found."
    
    # עיבוד התובנות בדיוק כמו בפעילויות
    mood_field_lower = mood_field.lower()
    header = f"## 💊 **Medication impact on {mood_field}**\n\n"
    
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
        
        # קביעת הכותרת/תווית להצגה ומסירת המילה "name_"
        if feature_type == "medication_name":
            # הסרת המילה "name_" מתחילת שם התרופה
            if feature_value.startswith("name_"):
                label = feature_value.replace("name_", "")
            else:
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
    
    # שלב הכל ביחד - רק ללא basic_insights
    combined_insights = pattern_insights + detailed_insights
    
    return combined_insights
ִִ##
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

    if not enriched_data:
        return "No mood data found after meals."

    df = pd.DataFrame(enriched_data)

    nutrients = {
        "proteins": "Protein",
        "carbohydrates": "Carbohydrates",
        "fats": "Fat",
        "dietaryFiber": "Fiber"
    }

    thresholds = {
        "proteins": 10,
        "carbohydrates": 20,
        "fats": 10,
        "dietaryFiber": 3
    }

    overall_avg = df["mood"].mean()
    mood_field_lower = mood_field.lower()

    insights = f"## 🍽️ Nutrition impact on {mood_field}\n\n"

    for key, label in nutrients.items():
        if key not in df.columns:
            continue

        threshold = thresholds[key]
        with_nutrient = df[df[key] >= threshold]
        without_nutrient = df[df[key] < threshold]

        if len(with_nutrient) < 2 or len(without_nutrient) < 2:
            continue

        with_avg = with_nutrient["mood"].mean()
        without_avg = without_nutrient["mood"].mean()
        diff = round(with_avg - without_avg, 2)

        if abs(diff) < 0.1:
            continue

        emoji = "🟢" if diff > 0 else "🔴"
        direction = "increases" if diff > 0 else "decreases"
        effect_str = f"{abs(diff):.1f}"

        insights += f"{emoji} {label}: {direction} {mood_field_lower} by {effect_str} on average\n\n"

    if insights.strip() == f"## 🍽️ Nutrition impact on {mood_field}":
        return "No significant nutrient patterns found."

    return insights


def symptom_analysis_summary(mood_field):
    """
    מציג סיכום של ניתוח הסימפטומים עם צבעים, בדומה לניתוח התרופות והפעילויות
    """
    if not translated_data_global:
        return "Please upload and process data first."
    
    # ניתוח מתקדם של דפוסי סימפטומים
    advanced_analysis = analyze_symptom_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        return advanced_analysis
    
    if not advanced_analysis:
        return "No symptom patterns found."
    
    # עיבוד התובנות
    mood_field_lower = mood_field.lower()
    header = f"## 🩺 **Symptom impact on {mood_field}**\n\n"
    
    green_insights = []
    red_insights = []
    neutral_insights = []
    
    for item in advanced_analysis:
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect):.1f}"  # עיגול לספרה אחת אחרי הנקודה
        
        # התווית היא שם הסימפטום בלי "type_"
        label = feature_value
        
        # קביעת כיוון ותו
        if abs(effect) < 0.05:
            line = f"⚫ **{label}**: no significant impact\n\n"
            neutral_insights.append(line)
        elif effect > 0:
            line = f"🟢 **{label}**: increases {mood_field_lower} by {effect_str} on average\n\n"
            green_insights.append(line)
        else:
            line = f"🔴 **{label}**: decreases {mood_field_lower} by {effect_str} on average\n\n"
            red_insights.append(line)
    
    # שילוב לפי סדר עדיפות
    combined_insights = header + "".join(green_insights + red_insights + neutral_insights)
    
    return combined_insights
    
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
