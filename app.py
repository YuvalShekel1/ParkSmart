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
import requests  # חדש
import time      #חדש

USDA_API_KEY ="BEQskSWfE4TbgTXy6GjTADB4ON7WX2ajidP3QPBq"
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"  #חדש

# Cache לחיפושים ב-API כדי למנוע קריאות מיותרות חדש
api_cache = {}  #חדש
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
    "דופיקאר 250 מ\"ג": "Dopicar 250 mg",
    "אזילקט 1 מ\"ג": "Azilect 1 mg",
    "מעדן סויה בטעמים": "Flavored soy pudding",
    "קפה": "Coffee",
    "חצי פיתה עם חמאת בוטנים": "Half pita with peanut butter",
    "פלפל ומלפפון": "Pepper and cucumber",
    "קערת קורנפלקס עם חלב סויה וצימוקים": "Bowl of cornflakes with soy milk and raisins",
    "קערת קורנפלקס עם חלב שקדים וצימוקים": "Bowl of cornflakes with almond milk and raisins",
    "סלמון עם פירה ואפונה": "Salmon with mashed potatoes and peas",
    "פיתה טחינה מלפפון עגבנייה ושניצל קטן": "Pita with tahini, cucumber, tomato and small schnitzel",
    "מעדן סויה אפרסק": "Peach soy pudding",
    "קערת קורנפלקס עם חלב סויה וצימוקים חדש": "Bowl of cornflakes with soy milk and raisins",
    "קפה אספרסו": "Espresso coffee",
    "רבע טוסט בייגלה": "Quarter of a pretzel toast",
    "אורז עם תבשיל טופו וכרובית": "Rice with tofu and cauliflower stew",
    "טוסט עם ביצה קשה": "Toast with hard-boiled egg",
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
    "אימון טנש": "Table Tennis Training",
    "אימון טנש קבוצתי": "Group Table Tennis Training",
    "משעה 2020 3 משחקים. הפסקה של 15 דקות לפני המשחקים": "From 8:20 PM, 3 games. 15-minute break before the games",
    "טנש": "Table Tennis",
    "טאקי": "Taki (card game)",
    "טורניר טנש": "Table Tennis tournament",
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

def usda_search_food(food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "pageSize": 5,  # במקום 1
        "api_key": USDA_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        # חפשי את הפריט הראשון שיש לו fdcId ונראה רלוונטי
        for food in results.get("foods", []):
            fdc_id = food.get("fdcId")
            description = food.get("description", "").lower()
            # נניח שאנחנו מעדיפים פירות טבעיים
            if "raw" in description or "fresh" in description or "apple" in description:
                return fdc_id

        # fallback – תחזירי את הראשון בכל מקרה
        if results.get("foods"):
            return results["foods"][0]["fdcId"]

        return None
    except Exception as e:
        print(f"USDA search error: {e}")
        return None
def usda_get_nutrition(fdc_id): #חדש
    if not USDA_API_KEY:
        return None
        
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {
        "api_key": USDA_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "foodNutrients" not in data:
            return None

        # חילוץ רכיבים תזונתיים
        nutrients = {}
        for nutrient in data.get("foodNutrients", []):
            # תמיכה בשני מבני נתונים
            nutrient_info = nutrient.get("nutrient", {})
            name = nutrient_info.get("name", "") or nutrient.get("name", "")
            value = nutrient.get("amount", 0) or nutrient.get("value", 0)
            
            if name and value is not None:
                nutrients[name] = value

        # מיפוי לערכים שאנחנו צריכים
        result = {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
        
        # חיפוש התאמות
        for nutrient_name, nutrient_value in nutrients.items():
            name_lower = nutrient_name.lower()
            value = float(nutrient_value or 0)
            
            if "protein" in name_lower and result["proteins"] == 0:
                result["proteins"] = round(value, 1)
            elif any(word in name_lower for word in ["total lipid", "fat"]) and result["fats"] == 0:
                result["fats"] = round(value, 1)
            elif "carbohydrate" in name_lower and "by difference" in name_lower and result["carbohydrates"] == 0:
                result["carbohydrates"] = round(value, 1)
            elif "fiber" in name_lower and "dietary" in name_lower and result["dietaryFiber"] == 0:
                result["dietaryFiber"] = round(value, 1)

        return result

    except Exception as e:
        return None
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


def extract_food_nutrition(food_name): #חדש
    # אם קיים במסד שלנו
    if food_name in nutrition_db:
        return nutrition_db[food_name]

    # נסה לחשב לפי חלקים מוכרים
    estimated = calculate_complex_meal_nutrition(food_name)
    if any(estimated.values()):
        return estimated

    # חיפוש ב־USDA אם לא נמצא
    fdc_id = usda_search_food(food_name)
    if fdc_id:
        api_result = usda_get_nutrition(fdc_id)
        if api_result:
            return api_result

    # אם כלום לא הצליח
    return {"proteins": 0, "fats": 0, "carbohydrates": 0, "dietaryFiber": 0}
    
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

# פונקציות ניתוח מתקדמות - עם רגרסיה לינארית
def analyze_activity_effect(df, target_column, preprocessor):
    try:
        df = df[df[target_column].notna()]

        if len(df) < 5:
            print("⚠️ Not enough data for regression.")
            return ["⚠️ Not enough data to analyze the effect of activities on " + target_column + "."]

        X = df[["activity_name", "duration", "intensity"]]
        y = df[target_column]

        model = make_pipeline(preprocessor, LinearRegression())
        model.fit(X, y)

        coefs = model.named_steps["linearregression"].coef_
        feature_names = model.named_steps["columntransformer"].get_feature_names_out()

        # חישוב ממוצע מוחלט של מקדמים של פעילות
        mean_abs_coef = np.mean([abs(c) for f, c in zip(feature_names, coefs) if "activity_name" in f])

        result = []
        for name, coef in zip(feature_names, coefs):
            if "activity_name" in name:
                feature_type = "activity_name"
                feature_value = name.split("_")[-1]
            elif "intensity" in name:
                feature_type = "intensity"
                feature_value = name.split("_")[-1]
            else:
                feature_type = "duration"
                feature_value = ""

            if feature_type == "activity_name" and mean_abs_coef:
                change = round(100 * (coef / mean_abs_coef), 1)
            else:
                change = round(coef, 4)

            result.append({
                "feature_type": feature_type,
                "feature_value": feature_value,
                "effect": change
            })

        # ===== רגרסיה מפורטת לפי פעילות =====
        try:
            df["duration_short"] = (df["duration"] < 30).astype(int)
            df["duration_medium"] = ((df["duration"] >= 30) & (df["duration"] < 60)).astype(int)
            df["duration_long"] = (df["duration"] >= 60).astype(int)

            for activity in df["activity_name"].unique():
                activity_df = df[df["activity_name"] == activity].copy()

                if len(activity_df) >= 3:
                    # משך זמן
                    if len(activity_df["duration_short"].unique()) > 1 or len(activity_df["duration_medium"].unique()) > 1 or len(activity_df["duration_long"].unique()) > 1:
                        X_duration = activity_df[["duration_medium", "duration_long"]]
                        y_duration = activity_df["mood_after"]
                        try:
                            duration_model = LinearRegression()
                            duration_model.fit(X_duration, y_duration)
                            for i, coef in enumerate(duration_model.coef_):
                                if abs(coef) >= 0.2:
                                    duration_desc = "between 30-60 minutes" if i == 0 else "more than 60 minutes"
                                    result.append({
                                        "feature_type": "detailed_duration",
                                        "feature_value": f"{activity} {duration_desc}",
                                        "effect": round(coef, 4)
                                    })
                        except:
                            pass

                    # עצימות
                    if len(activity_df["intensity"].unique()) > 1:
                        try:
                            intensity_dummies = pd.get_dummies(activity_df["intensity"], prefix="intensity")
                            intensity_data = pd.concat([intensity_dummies, activity_df["mood_after"]], axis=1)
                            X_intensity = intensity_data.drop("mood_after", axis=1)
                            y_intensity = intensity_data["mood_after"]
                            intensity_model = LinearRegression()
                            intensity_model.fit(X_intensity, y_intensity)
                            for intensity_name, coef in zip(X_intensity.columns, intensity_model.coef_):
                                if abs(coef) >= 0.2:
                                    intensity_value = intensity_name.split("_")[-1]
                                    result.append({
                                        "feature_type": "detailed_intensity",
                                        "feature_value": f"{activity} with {intensity_value} intensity",
                                        "effect": round(coef, 4)
                                    })
                        except:
                            pass

                    # שילובים
                    if len(activity_df) >= 4 and len(activity_df["intensity"].unique()) > 1:
                        try:
                            combined_features = pd.DataFrame()
                            intensity_dummies = pd.get_dummies(activity_df["intensity"], prefix="intensity")
                            for duration_type in ["duration_short", "duration_medium", "duration_long"]:
                                for intensity_col in intensity_dummies.columns:
                                    col_name = f"{duration_type}_{intensity_col}"
                                    combined_features[col_name] = activity_df[duration_type] * intensity_dummies[intensity_col]

                            if combined_features.shape[1] > 0:
                                X_combined = combined_features
                                y_combined = activity_df["mood_after"]
                                combined_model = LinearRegression()
                                combined_model.fit(X_combined, y_combined)
                                for feature_name, coef in zip(X_combined.columns, combined_model.coef_):
                                    if abs(coef) >= 0.2:
                                        parts = feature_name.split("_")
                                        duration_type = parts[1]
                                        intensity_value = parts[-1]
                                        duration_desc = "less than 30 minutes" if duration_type == "short" else "between 30-60 minutes" if duration_type == "medium" else "more than 60 minutes"
                                        result.append({
                                            "feature_type": "detailed_combo",
                                            "feature_value": f"{activity} {duration_desc} with {intensity_value} intensity",
                                            "effect": round(coef, 4)
                                        })
                        except:
                            pass
        except Exception as e:
            print(f"Error in detailed activity analysis: {str(e)}")
            pass

        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)
        return result

    except Exception as e:
        return [f"Error in activity pattern analysis: {str(e)}"]


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
                "effect": round(coef, 4)
            })

        # ניתוח השפעת מרווחי זמן - עם רגרסיה לינארית
        if "time_diff_hours" in df.columns and len(df) >= 4:
            try:
                # יצירת קטגוריות של חלונות זמן
                df["time_0_1"] = (df["time_diff_hours"] <= 1).astype(int)
                df["time_1_2"] = ((df["time_diff_hours"] > 1) & (df["time_diff_hours"] <= 2)).astype(int)
                df["time_2_4"] = ((df["time_diff_hours"] > 2) & (df["time_diff_hours"] <= 4)).astype(int)
                df["time_4_plus"] = (df["time_diff_hours"] > 4).astype(int)
                
                # נבדוק את ההשפעה של כל תרופה בחלונות זמן שונים
                for med in df["medication_name"].unique():
                    med_data = df[df["medication_name"] == med].copy()
                    
                    # שינוי כאן: מינימום 2 במקום 3
                    if len(med_data) >= 2:
                        # רגרסיה לינארית על חלונות הזמן
                        X_time = med_data[["time_0_1", "time_1_2", "time_2_4", "time_4_plus"]]
                        y_time = med_data["mood_after"]
                        
                        # בדוק שיש שונות בנתונים
                        if X_time.std().sum() > 0:
                            try:
                                time_model = LinearRegression()
                                time_model.fit(X_time, y_time)
                                
                                # חילוץ המקדמים
                                time_windows = ["0-1 hour", "1-2 hours", "2-4 hours", "4+ hours"]
                                for i, coef in enumerate(time_model.coef_):
                                    time_label = time_windows[i]
                                    if time_label == "4+ hours":
                                        continue  # מדלג על תובנות של 4+ שעות
                                    if abs(coef) >= 0.2:
                                        result.append({
                                            "feature_type": "time_window",
                                            "feature_value": f"{med} within {time_label}",
                                            "effect": round(coef, 4)
                                        })
                            except:
                                pass
            except Exception as e:
                print(f"Error in time window analysis: {str(e)}")
                pass

        # ניתוח רצפי תרופות - עם רגרסיה לינארית
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
                    
                    # יצירת רצפים
                    sequences = []
                    for i in range(len(sorted_meds) - 1):
                        first = sorted_meds[i]["name"]
                        second = sorted_meds[i + 1]["name"]
                        sequences.append(f"{first} → {second}")
                    
                    # מצא דיווחים על מצב רוח לאחר הרצף
                    last_med_time = sorted_meds[-1]["time"]
                    moods_after = mood_df[mood_df["date"] > last_med_time]
                    same_day_end = pd.Timestamp(date + " 23:59:59")
                    same_day_moods = moods_after[moods_after["date"] <= same_day_end]
                    
                    if not same_day_moods.empty:
                        avg_mood = same_day_moods["severity"].mean()
                        for sequence in sequences:
                            sequence_data.append({
                                "sequence": sequence,
                                "mood": avg_mood,
                                "date": date
                            })
            
            # רגרסיה לינארית על רצפי תרופות
            if len(sequence_data) >= 2:  # מינימום 2 תצפיות
                seq_df = pd.DataFrame(sequence_data)
                
                # בדוק אם יש מספיק רצפים שונים
                if len(seq_df["sequence"].unique()) >= 2:
                    # יצירת משתנים דמי לרצפים
                    X_seq = pd.get_dummies(seq_df["sequence"], drop_first=False)
                    y_seq = seq_df["mood"]
                    
                    # רגרסיה לינארית
                    seq_model = LinearRegression()
                    seq_model.fit(X_seq, y_seq)
                    
                    # חילוץ המקדמים
                    for seq_name, coef in zip(X_seq.columns, seq_model.coef_):
                        if abs(coef) >= 0.2:  # רק מקדמים משמעותיים
                            result.append({
                                "feature_type": "medication_sequence",
                                "feature_value": seq_name,
                                "effect": round(coef, 4)
                            })
        except Exception as e:
            print(f"Error in medication sequence analysis: {str(e)}")
            pass

        # מיון התוצאות לפי גודל ההשפעה (מוחלט)
        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)

        return result
    except Exception as e:
        return f"Error in medication pattern analysis: {str(e)}"

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

        # נבנה דאטה פריים עם תאריכים ומצבי רוח
        date_to_mood = {}
        for _, row in mood_df.iterrows():
            date_str = row["date"].date().isoformat()
            if date_str not in date_to_mood:
                date_to_mood[date_str] = []
            date_to_mood[date_str].append(row["severity"])
        
        # ממוצע מצב רוח לכל יום
        daily_mood = {date: sum(moods)/len(moods) for date, moods in date_to_mood.items()}
        
        # נכין נתונים לרגרסיה לינארית - עבור כל יום, נציין אם היה כל סימפטום
        symptom_types = symptom_df["symptom_type"].unique()
        
        regression_data = []
        for date_str, avg_mood in daily_mood.items():
            date_obj = pd.Timestamp(date_str).date()
            
            # מצא את כל הסימפטומים של היום הזה
            day_symptoms = symptom_df[symptom_df["date"].dt.date == date_obj]
            
            # יצירת רשומה עם משתנים דמי לכל סימפטום
            record = {"date": date_str, "mood": avg_mood}
            
            for symptom in symptom_types:
                # בדוק אם הסימפטום הזה דווח ביום זה
                has_symptom = (day_symptoms["symptom_type"] == symptom).any()
                record[f"symptom_{symptom}"] = int(has_symptom)
            
            regression_data.append(record)
        
        # בדוק שיש מספיק נתונים
        if len(regression_data) < 3:
            return "Not enough daily records for symptom analysis."
            
        # יצירת דאטה פריים
        reg_df = pd.DataFrame(regression_data)
        
        # הכנת נתונים לרגרסיה
        X_cols = [col for col in reg_df.columns if col.startswith("symptom_")]
        
        # בדוק שיש לפחות 2 סימפטומים שונים
        if len(X_cols) < 2:
            return "Not enough different symptoms for analysis."
            
        X = reg_df[X_cols]
        y = reg_df["mood"]
        
        # רגרסיה לינארית
        model = LinearRegression()
        model.fit(X, y)
        
        # חילוץ המקדמים
        result = []
        for symptom_col, coef in zip(X_cols, model.coef_):
            # הוצאת שם הסימפטום
            symptom_name = symptom_col.replace("symptom_", "")
            
            result.append({
                "feature_type": "symptom_type",
                "feature_value": symptom_name,
                "effect": round(coef, 4),
                "count": X[symptom_col].sum()  # מספר הימים עם הסימפטום
            })
        
        # מיון התוצאות לפי גודל ההשפעה (מוחלט)
        result.sort(key=lambda x: abs(x.get("effect", 0)), reverse=True)

        return result
    except Exception as e:
        return f"Error in symptom pattern analysis: {str(e)}"
        
def determine_colors(effect, mood_field):
    """
    Helper function to determine if an effect is positive or negative
    based on the mood field.
    """
    mood_field_lower = mood_field.lower()
    if "mood" in mood_field_lower or "well-being" in mood_field_lower:
        # For mood/well-being, positive effect is good (increases)
        is_positive = effect > 0
        is_negative = effect < 0
    elif "parkinson" in mood_field_lower or "symptom" in mood_field_lower:
        # For Parkinson's/symptoms, negative effect (decrease) is good
        is_positive = effect < 0
        is_negative = effect > 0
    else:
        # Default to positive effect being good if not specified
        is_positive = effect > 0
        is_negative = effect < 0
    return is_positive, is_negative

def nutrition_analysis_summary(mood_field):
    """
    ניתוח השפעת התזונה על מצב הרוח/פרקינסון באמצעות רגרסיה לינארית
    עם הפרדה בין ערכים תזונתיים בסיסיים למאכלים ספציפיים
    """
    global translated_data_global 
    if not translated_data_global:
        return "Please upload and process data first."
    
    try:
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

            # חיפוש מצב רוח אחרי האוכל באותו היום
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

        # בדיקה אם יש מספיק נתונים
        if len(df) < 3:
            return "Not enough nutrition-mood data pairs for analysis."
            
        # רגרסיה לינארית עבור רכיבים תזונתיים
        nutrient_result = []
        
        # הכנת הנתונים לרגרסיה
        nutrient_cols = [col for col in df.columns if col in nutrients.keys()]
        
        if not nutrient_cols:
            return "No nutritional data available."
            
        X_nutrients = df[nutrient_cols]
        y_mood = df["mood"]
        
        # בדיקה שיש שונות בנתונים
        if X_nutrients.std().sum() > 0:
            # רגרסיה לינארית
            nutrient_model = LinearRegression()
            nutrient_model.fit(X_nutrients, y_mood)
            
            # חילוץ המקדמים
            for nutrient_col, coef in zip(nutrient_cols, nutrient_model.coef_):
                is_positive, is_negative = determine_colors(coef, mood_field)
                nutrient_result.append({
                    "feature_type": "nutrient",
                    "feature_value": nutrients[nutrient_col],
                    "effect": round(coef, 4),
                    "is_positive": is_positive,
                    "is_negative": is_negative,
                    "is_significant": abs(coef) >= 0.05 # Add significance for sorting
                })
        
        # ניתוח מזונות ספציפיים עם רגרסיה לינארית
        food_result = []
        
        # בדיקה אילו מזונות מופיעים מספיק פעמים
        common_foods = df['food'].value_counts()
        common_foods = common_foods[common_foods >= 2]
        
        if len(common_foods) >= 2: # צריך לפחות 2 סוגי מזון שונים
            # יצירת משתנים דמי למזונות
            food_dummies = pd.get_dummies(df['food'], prefix='food')
            
            # רגרסיה לינארית
            food_model = LinearRegression()
            food_model.fit(food_dummies, y_mood)
            
            # חילוץ המקדמים
            for food_col, coef in zip(food_dummies.columns, food_model.coef_):
                is_positive, is_negative = determine_colors(coef, mood_field)
                food_name = food_col.replace('food_', '')
                food_result.append({
                    "feature_type": "specific_food",
                    "feature_value": food_name,
                    "effect": round(coef, 4),
                    "is_positive": is_positive,
                    "is_negative": is_negative,
                    "is_significant": abs(coef) >= 0.1 # Add significance for sorting
                })
        
        # Custom sorting key for nutrient_result and food_result
        def sort_key(item):
            if item["is_positive"] and item["is_significant"]:
                return (0, -abs(item.get("effect", 0)))  # Green, then by effect magnitude (desc)
            elif item["is_negative"] and item["is_significant"]:
                return (1, -abs(item.get("effect", 0)))  # Red, then by effect magnitude (desc)
            else:
                return (2, 0)  # Black (no significant impact)

        nutrient_result.sort(key=sort_key)
        food_result.sort(key=sort_key)
            
        mood_field_lower = mood_field.lower()
            
        # --- בניית ה-HTML עבור רכיבים תזונתיים ---
        nutrient_insights_lines_html = []
        for item in nutrient_result:
            feature_value = item.get("feature_value", "")
            effect = item.get("effect")
            effect_str = f"{abs(effect)/5*100:.1f}%"
            is_positive = item["is_positive"]
            is_negative = item["is_negative"]
            is_significant = item["is_significant"]
            
            # שימוש ביוניקוד וב-<span> עם סגנון ישיר לצבעים במקום Markdown
            if not is_significant:
                line_html = f"<p>&#x26AB; <strong>{feature_value}</strong>: no significant impact</p>" # Black circle
            elif is_positive:
                direction = "increases" if effect > 0 else "decreases" # This direction is for actual effect value, not necessarily "good" or "bad"
                line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{feature_value}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
            else:  # is_negative
                direction = "increases" if effect > 0 else "decreases"
                line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{feature_value}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
            
            nutrient_insights_lines_html.append(line_html)
            
        nutrient_insights_html_section = f"""
        <h2>🍴 <strong>Nutrition impact on {mood_field}</strong></h2>
        {"".join(nutrient_insights_lines_html)}
        """
            
        # --- בניית ה-HTML עבור מזונות ספציפיים ---
        food_insights_html_section = ""
        if food_result:
            food_insights_lines_html = []
            for item in food_result:
                feature_value = item.get("feature_value", "")
                effect = item.get("effect")
                effect_str = f"{abs(effect)/5*100:.1f}%"
                is_positive = item["is_positive"]
                is_negative = item["is_negative"]
                is_significant = item["is_significant"]
                
                if not is_significant:
                    line_html = f"<p>&#x26AB; <strong>{feature_value}</strong>: no significant impact</p>"
                elif is_positive:
                    direction = "increases" if effect > 0 else "decreases"
                    line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{feature_value}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>"
                else:  # is_negative
                    direction = "increases" if effect > 0 else "decreases"
                    line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{feature_value}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>"
                    
                food_insights_lines_html.append(line_html)
                
            food_insights_html_section = f"""
            <h2>🍽️ Detailed Food Patterns</h2>
            {"".join(food_insights_lines_html)}
            """
            
        # --- שילוב כל החלקים במבנה HTML סופי ---
        # בדיקה אם יש תוכן משמעותי לפני בניית ה-HTML
        if not nutrient_insights_lines_html and not food_insights_lines_html:
            return "No significant nutrient or food patterns found."

        # זהו המבנה הסופי שיוחזר כ-HTML
        final_html_output = f"""
        <div id="nutrition-analysis-container" class="svelte-vuh1yp">
            <div class="prose svelte-lag733" data-testid="markdown" dir="ltr" style="">
                <span class="md svelte-7ddecg prose">
                    <div class="column-content">
                        {nutrient_insights_html_section}
                    </div>
                    <div class="column-content">
                        {food_insights_html_section}
                    </div>
                </span>
            </div>
        </div>
        """
        return final_html_output
        
    except Exception as e:
        return f"Error in nutrition analysis: {str(e)}"
# פונקציה לקביעת הצבעים לפי סוג שדה המצב
def determine_colors(effect, mood_field):
    """
    קובע את הצבע והכיוון לפי סוג שדה המצב:
    - עבור My Mood: עלייה = חיובי (ירוק), ירידה = שלילי (אדום)
    - עבור Parkinson's State ו-Physical State: עלייה = שלילי (אדום), ירידה = חיובי (ירוק)
    """
    if mood_field in ["Parkinson's State", "Physical State"]:
        # היפוך הצבעים - ירידה במצב הפרקינסון או הפיזי היא חיובית
        return effect < 0, effect > 0
    else:
        # השארת הצבעים כמו שהם - עלייה במצב הרוח היא חיובית
        return effect > 0, effect < 0

def activity_analysis_summary(mood_field):
    global translated_data_global
    if not translated_data_global:
        return "Please upload and process data first."

    advanced_analysis = analyze_activity_patterns(translated_data_global, mood_field)

    if isinstance(advanced_analysis, str):
        return advanced_analysis

    if not advanced_analysis:
        return "No patterns found."

    mood_field_lower = mood_field.lower()

    # Lists to store HTML lines for basic insights
    basic_green_html = []
    basic_red_html = []
    basic_neutral_html = []
    
    # Lists to store HTML lines for detailed insights
    detailed_green_html = []
    detailed_red_html = []

    for item in advanced_analysis:
        feature_type = item.get("feature_type", "")
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect)/5*100:.1f}%"
        # Determine label for display
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

        is_positive, is_negative = determine_colors(effect, mood_field)
        direction = "increases" if effect > 0 else "decreases"

        # Construct HTML line
        if abs(effect) < 0.05:
            line_html = f"<p>&#x26AB; <strong>{label}</strong>: no significant impact</p>" # Black circle
            basic_neutral_html.append(line_html)
        elif is_positive:
            if feature_type in ["detailed_duration", "detailed_intensity", "detailed_combo"]:
                line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{label}</strong> {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
                detailed_green_html.append(line_html)
            else:
                line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
                basic_green_html.append(line_html)
        else: # is_negative
            if feature_type in ["detailed_duration", "detailed_intensity", "detailed_combo"]:
                line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{label}</strong> {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
                detailed_red_html.append(line_html)
            else:
                line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
                basic_red_html.append(line_html)

    # Combine basic insights into an HTML section
    activity_insights_html_section = f"""
    <h2>🏃 <strong>Activity impact on {mood_field}</strong></h2>
    {"".join(basic_green_html + basic_red_html + basic_neutral_html)}
    """

    # Combine detailed insights into an HTML section
    detailed_activity_insights_html_section = ""
    if detailed_green_html or detailed_red_html:
        detailed_activity_insights_html_section = f"""
        <h2>🏋️ Detailed Activity Patterns</h2>
        {"".join(detailed_green_html + detailed_red_html)}
        """

    # Final HTML output structure with two columns
    if not basic_green_html and not basic_red_html and not basic_neutral_html and not detailed_green_html and not detailed_red_html:
        return "No significant activity patterns found."

    final_html_output = f"""
    <div id="activity-analysis-container" class="svelte-vuh1yp">
        <div class="prose svelte-lag733" data-testid="markdown" dir="ltr" style="">
            <span class="md svelte-7ddecg prose">
                <div class="column-content">
                    {activity_insights_html_section}
                </div>
                <div class="column-content">
                    {detailed_activity_insights_html_section}
                </div>
            </span>
        </div>
    </div>
    """
    return final_html_output

def medication_analysis_summary(mood_field):
    """
    מציג סיכום של ניתוח התרופות עם צבעים, בדומה לניתוח הפעילויות
    """
    if not translated_data_global:
        return "Please upload and process data first."
    
    # ניתוח מתקדם של דפוסים בתרופות
    advanced_analysis = analyze_medication_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        return advanced_analysis
    
    if not advanced_analysis:
        return "No medication patterns found."
    
    # עיבוד התובנות בדיוק כמו בפעילויות
    mood_field_lower = mood_field.lower()
    
    basic_green_html = []
    basic_red_html = []
    basic_neutral_html = []
    
    # תובנות דפוסים מפורטים של חלונות זמן ורצפי תרופות
    detailed_green_html = []
    detailed_red_html = []
    
    for item in advanced_analysis:
        feature_type = item.get("feature_type", "")
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect)/5*100:.1f}%"
        
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
        
        # קביעת כיוון והצבע לפי סוג שדה המצב
        is_positive, is_negative = determine_colors(effect, mood_field)
        direction = "increases" if effect > 0 else "decreases"
        
        # Construct HTML line
        if abs(effect) < 0.05:
            line_html = f"<p>&#x26AB; <strong>{label}</strong>: no significant impact</p>" # Black circle
            basic_neutral_html.append(line_html)
        elif is_positive:
            if feature_type in ["time_window", "medication_sequence"]:
                line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{label}</strong> {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
                detailed_green_html.append(line_html)
            else:
                line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
                basic_green_html.append(line_html)
        else: # is_negative
            if feature_type in ["time_window", "medication_sequence"]:
                line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{label}</strong> {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
                detailed_red_html.append(line_html)
            else:
                line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
                basic_red_html.append(line_html)
    
    # Combine basic insights into an HTML section
    medication_insights_html_section = f"""
    <h2>💊 <strong>Medication impact on {mood_field}</strong></h2>
    {"".join(basic_green_html + basic_red_html + basic_neutral_html)}
    """

    # Combine detailed insights into an HTML section
    detailed_medication_insights_html_section = ""
    if detailed_green_html or detailed_red_html:
        detailed_medication_insights_html_section = f"""
        <h2>💉 Detailed Medication Patterns</h2>
        {"".join(detailed_green_html + detailed_red_html)}
        """

    # Final HTML output structure with two columns
    if not basic_green_html and not basic_red_html and not basic_neutral_html and not detailed_green_html and not detailed_red_html:
        return "No significant medication patterns found."

    final_html_output = f"""
    <div id="medication-analysis-container" class="svelte-vuh1yp">
        <div class="prose svelte-lag733" data-testid="markdown" dir="ltr" style="">
            <span class="md svelte-7ddecg prose">
                <div class="column-content">
                    {medication_insights_html_section}
                </div>
                <div class="column-content">
                    {detailed_medication_insights_html_section}
                </div>
            </span>
        </div>
    </div>
    """
    return final_html_output

def determine_colors(effect, mood_field):
    """
    Helper function to determine if an effect is positive or negative
    based on the mood field.
    """
    mood_field_lower = mood_field.lower()
    if "mood" in mood_field_lower or "well-being" in mood_field_lower:
        # For mood/well-being, positive effect is good (increases)
        is_positive = effect > 0
        is_negative = effect < 0
    elif "parkinson" in mood_field_lower or "symptom" in mood_field_lower:
        # For Parkinson's/symptoms, negative effect (decrease) is good
        is_positive = effect < 0
        is_negative = effect > 0
    else:
        # Default to positive effect being good if not specified
        is_positive = effect > 0
        is_negative = effect < 0
    return is_positive, is_negative


def symptom_analysis_summary(mood_field):
    """
    מציג סיכום של ניתוח הסימפטומים עם צבעים, בדומה לניתוח התרופות והפעילויות
    """
    if not translated_data_global:
        return "Please upload and process data first."
        
    # ניתוח מתקדם של דפוסים בסימפטומים
    advanced_analysis = analyze_symptom_patterns(translated_data_global, mood_field)
    
    if isinstance(advanced_analysis, str):
        return advanced_analysis
    
    if not advanced_analysis:
        return "No symptom patterns found."
        
    # עיבוד התובנות - כל התובנות יבנו במקטע HTML אחד
    mood_field_lower = mood_field.lower()
    
    # צור רשימה שתכיל מילונים עבור כל תובנה, כולל סוג הצבע למיון
    insights_with_color_info = []
    
    for item in advanced_analysis:
        feature_value = item.get("feature_value", "")
        effect = item.get("effect")
        effect_str = f"{abs(effect)/5*100:.1f}%" # עיגול לספרה אחת אחרי הנקודה
        
        # התווית היא שם הסימפטום
        label = feature_value
        
        # קביעת כיוון והצבע לפי סוג שדה המצב
        is_positive, is_negative = determine_colors(effect, mood_field)
        direction = "increases" if effect > 0 else "decreases"
        
        line_html = ""
        color_priority = 0 # 0 for green, 1 for red, 2 for black

        if abs(effect) < 0.05:
            line_html = f"<p>&#x26AB; <strong>{label}</strong>: no significant impact</p>" # Black circle
            color_priority = 2 # Set priority for black
        elif is_positive:
            line_html = f"<p><span style='color: green;'>&#x1F7E2;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Green circle
            color_priority = 0 # Set priority for green
        else: # is_negative
            line_html = f"<p><span style='color: red;'>&#x1F534;</span> <strong>{label}</strong>: {direction} {mood_field_lower} by {effect_str} on average</p>" # Red circle
            color_priority = 1 # Set priority for red
        
        insights_with_color_info.append({
            "html": line_html,
            "priority": color_priority,
            "effect_abs": abs(effect) # גם נמיין לפי גודל ההשפעה בתוך כל קבוצת צבע
        })
    
    # מיון התובנות לפי סדר קדימות הצבעים: ירוק (0), אדום (1), שחור (2)
    # ובתוך כל קבוצת צבע, מיין לפי גודל ההשפעה בסדר יורד
    insights_with_color_info.sort(key=lambda x: (x["priority"], -x["effect_abs"]))

    # כעת, בנה את רשימת מחרוזות ה-HTML מהרשימה הממוינת
    all_insights_html_lines = [item["html"] for item in insights_with_color_info]

    # בניית החלק הראשי של ה-HTML (כל התובנות בעמודה אחת)
    main_symptom_insights_html_section = f"""
    <h2>🩺 <strong>Symptom impact on {mood_field}</strong></h2>
    {"".join(all_insights_html_lines)}
    """ if all_insights_html_lines else ""

    # Handle cases where no patterns at all are found
    if not all_insights_html_lines:
        return "No significant symptom patterns found."

    # Final HTML output structure - Use only ONE column-content div
    final_html_output = f"""
    <div id="symptom-analysis-container" class="svelte-vuh1yp">
        <div class="prose svelte-lag733" data-testid="markdown" dir="ltr" style="">
            <span class="md svelte-7ddecg prose">
                <div class="column-content">  
                    {main_symptom_insights_html_section}
                </div>
            </span>
        </div>
    </div>
    """
    return final_html_output
    
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
# הוספת CSS מותאם אישית
custom_css = """
.gradio-container {
    background-color: #ffffff !important;
    font-size: 16px !important;
}
button.primary {
    background-color: #a6cee3 !important;
    border-color: #a6cee3 !important;
    color: #000000 !important;
}
button.primary:hover {
    background-color: #8bb8d9 !important;
    border-color: #8bb8d9 !important;
    color: #000000 !important;
}
.markdown {
    font-size: 18px !important;
}
#component-3 .wrap.svelte-12ioyct {
    flex-direction: row !important;
    gap: 8px;
    align-items: center !important;
    justify-content: center !important;
    min-height: auto !important;
    height: auto !important;
    padding: 3px !important;
}
#component-2 button#component-4 {
    padding: 9px calc(16px);
}
#component-7 .empty.svelte-1oiin9d.large.unpadded_box {
    min-height: 87px;
}
div#component-5 {
    flex-wrap: nowrap !important;
    gap: 22px !important;
    box-sizing: border-box !important;
}
div#component-6 {
    height: fit-content !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
}
div#component-7 {
    flex-grow: 1 !important;
    min-width: 50% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
div#component-3 .icon-wrap.svelte-12ioyct {
    width: 20px;
    margin-bottom: var(--spacing-lg);
    margin-bottom: 3px;
}
div#component-3 .wrap.svelte-12ioyct {
    font-size: 14px !important;
    white-space: nowrap !important;
    gap: 4px !important;
}
button.svelte-edrmkl.center.boundedheight.flex {
    min-height: 41px !important;
}
div#component-3, div#component-7 {
    border-style: solid !important;
    border-color: #d1d5db !important;
    border-width: 1px !important;
    border-radius: var(--radius-sm) !important;
}
.form.svelte-633qhp {
    border: 1px solid #d1d5db !important;
}
#nutrition-analysis-container span.md.svelte-7ddecg.prose {
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
    box-sizing: border-box !important;
    gap: 80px !important;
}
#medication-analysis-container span.md.svelte-7ddecg.prose {
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
    box-sizing: border-box !important;
    gap: 80px !important;
}
#activity-analysis-container span.md.svelte-7ddecg.prose {
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
    box-sizing: border-box !important;
    gap: 80px !important;
}
#symptom-analysis-container span.md.svelte-7ddecg.prose {
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    width: 100% !important;
    box-sizing: border-box !important;
    gap: 80px !important;
}
"""
with gr.Blocks(title="Parkinson's Health Pattern Analysis", css=custom_css) as app:
    gr.Markdown("# 📈 Parkinson's Health Pattern Analysis")

    with gr.Row():
        file_input = gr.File(label="Upload JSON File")
        upload_button = gr.Button("Upload and Process", variant="primary", size="lg")
    with gr.Row():
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
            activity_output = gr.HTML(label="Activity Insights")
        
        with gr.TabItem("💊 Medication Analysis"):
            medication_button = gr.Button("Analyze Medication Patterns", variant="primary")
            medication_output = gr.HTML(label="Medication Insights")

        with gr.TabItem("🩺 Symptom Analysis"):
            symptom_button = gr.Button("Analyze Symptom Patterns", variant="primary")
            symptom_output = gr.HTML(label="Symptom Insights")
            
        with gr.TabItem("🍽️ Nutrition Analysis"):
            nutrition_button = gr.Button("Analyze Nutrition Patterns", variant="primary")
            nutrition_output = gr.HTML(label="Nutrition Insights")    

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
