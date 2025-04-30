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
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
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
        
        # Make sure all needed keys exist, including the 'feelings' field
        keys_to_check = ["nutritions", "activities", "medications", "symptoms", "medicines", "feelings"]
        for key in keys_to_check:
            if key not in original_full_json:
                original_full_json[key] = []
        
        # Merge 'feelings' into 'symptoms' if needed
        if "feelings" in original_full_json and isinstance(original_full_json["feelings"], list):
            # Add each feeling to symptoms with the same structure
            for feeling in original_full_json["feelings"]:
                if isinstance(feeling, dict):
                    # Already in the right format
                    original_full_json["symptoms"].append(feeling)
                elif isinstance(feeling, list):
                    # List of feelings
                    original_full_json["symptoms"].extend(feeling)
        
        # Convert date fields to standard format if they exist
        date_keys = ["nutritions", "activities", "medications", "symptoms", "medicines"]
        for key in date_keys:
            if key in original_full_json:
                for item in original_full_json[key]:
                    if "dateTaken" in item:
                        item["date"] = item["dateTaken"]
                    # Ensure dates are in consistent format
                    if "date" in item:
                        try:
                            item["date"] = pd.to_datetime(item["date"]).isoformat()
                        except:
                            pass
        
        # Make sure medicine/medications are consistent
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
                            # Update nutritional values for food items
                            if key == "nutritions" and "foodName" in item:
                                food_name = item["foodName"]
                                item["nutritionalValues"] = extract_food_nutrition(food_name)
                
                # Translate the section
                original_full_json[key] = translate_value(section)

        translated_data_global = original_full_json

        # Save the processed data
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)

        return output_path, "✅ File processed successfully! All nutritional values have been updated and data has been fully translated."
    except Exception as e:
        return None, f"❌ Error processing: {str(e)}"

# New function: Extract time patterns
def extract_time_patterns(data, field_type, mood_field):
    """Extract time-based patterns for a given field and mood"""
    if not data:
        return {}
    
    try:
        # Get relevant data points
        mood_data = []
        if "symptoms" in data:
            for item in data["symptoms"]:
                if "date" in item and "type" in item and item.get("type") == mood_field:
                    mood_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "severity": item.get("severity", 0)
                    })
        
        # Get the selected field data
        field_data = []
        if field_type in data:
            for item in data[field_type]:
                if "date" in item:
                    field_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "data": item
                    })
        
        if not mood_data or not field_data:
            return {}
        
        # Create DataFrames
        mood_df = pd.DataFrame(mood_data)
        field_df = pd.DataFrame(field_data)
        
        # Extract time of day pattern
        mood_df["hour"] = mood_df["date"].dt.hour
        field_df["hour"] = field_df["date"].dt.hour
        
        # Group by time of day
        morning_mood = mood_df[mood_df["hour"] < 12]["severity"].mean()
        afternoon_mood = mood_df[(mood_df["hour"] >= 12) & (mood_df["hour"] < 18)]["severity"].mean()
        evening_mood = mood_df[mood_df["hour"] >= 18]["severity"].mean()
        
        morning_count = len(field_df[field_df["hour"] < 12])
        afternoon_count = len(field_df[(field_df["hour"] >= 12) & (field_df["hour"] < 18)])
        evening_count = len(field_df[field_df["hour"] >= 18])
        
        return {
            "morning_mood": morning_mood,
            "afternoon_mood": afternoon_mood,
            "evening_mood": evening_mood,
            "morning_count": morning_count,
            "afternoon_count": afternoon_count,
            "evening_count": evening_count
        }
    except Exception as e:
        print(f"Error in time pattern extraction: {str(e)}")
        return {}

# New function: Extract nutritional patterns using clustering
def analyze_nutrition_clusters(data, mood_field):
    if not data or "nutritions" not in data or "symptoms" not in data:
        return "Not enough data for clustering analysis."
    
    try:
        # Prepare data for clustering
        nutrition_data = []
        for item in data["nutritions"]:
            if "nutritionalValues" in item and "date" in item:
                nutrition_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "proteins": item["nutritionalValues"].get("proteins", 0),
                    "fats": item["nutritionalValues"].get("fats", 0),
                    "carbs": item["nutritionalValues"].get("carbohydrates", 0),
                    "fiber": item["nutritionalValues"].get("dietaryFiber", 0),
                    "food_name": item.get("foodName", "")
                })
        
        # Get mood data
        mood_data = []
        for item in data["symptoms"]:
            if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item.get("severity", 0)
                })
        
        if len(nutrition_data) < 5 or len(mood_data) < 5:
            return "Not enough data points for clustering analysis."
        
        # Create DataFrames
        nutrition_df = pd.DataFrame(nutrition_data)
        mood_df = pd.DataFrame(mood_data)
        
        # Join nutrition with mood data (within 3 hours)
        merged_data = []
        for _, nutr_row in nutrition_df.iterrows():
            nutr_date = nutr_row["date"]
            # Find closest mood measurement
            closest_mood = None
            min_diff = float('inf')
            
            for _, mood_row in mood_df.iterrows():
                mood_date = mood_row["date"]
                diff = abs((nutr_date - mood_date).total_seconds() / 3600)  # Hours
                
                if diff < min_diff and diff <= 3:  # Within 3 hours
                    min_diff = diff
                    closest_mood = mood_row["severity"]
            
            if closest_mood is not None:
                nutr_dict = nutr_row.to_dict()
                nutr_dict["mood"] = closest_mood
                merged_data.append(nutr_dict)
        
        if len(merged_data) < 5:
            return "Not enough matched data points for clustering."
        
        analysis_df = pd.DataFrame(merged_data)
        
        # Prepare features for clustering
        features = ["proteins", "fats", "carbs", "fiber"]
        X = analysis_df[features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters (2-4)
        silhouette_scores = []
        for k in range(2, 5):
            if len(X_scaled) > k:  # Make sure we have enough data points
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append((k, silhouette_avg))
        
        if not silhouette_scores:
            return "Not enough data for meaningful clusters."
        
        # Choose best number of clusters
        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
        
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        analysis_df["cluster"] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_results = []
        for cluster_id in range(best_k):
            cluster_data = analysis_df[analysis_df["cluster"] == cluster_id]
            avg_mood = cluster_data["mood"].mean()
            avg_proteins = cluster_data["proteins"].mean()
            avg_fats = cluster_data["fats"].mean()
            avg_carbs = cluster_data["carbs"].mean()
            avg_fiber = cluster_data["fiber"].mean()
            
            cluster_results.append({
                "cluster_id": cluster_id,
                "count": len(cluster_data),
                "avg_mood": round(avg_mood, 2),
                "nutrition_profile": {
                    "proteins": round(avg_proteins, 1),
                    "fats": round(avg_fats, 1),
                    "carbs": round(avg_carbs, 1),
                    "fiber": round(avg_fiber, 1)
                },
                "common_foods": Counter(cluster_data["food_name"]).most_common(3)
            })
        
        # Sort clusters by mood (best to worst)
        cluster_results.sort(key=lambda x: x["avg_mood"])
        
        return cluster_results
    except Exception as e:
        return f"Error in cluster analysis: {str(e)}"

# New function: Find association rules between medications and symptoms
def analyze_medication_patterns(data, mood_field):
    if not data or "medications" not in data or "symptoms" not in data:
        return "Not enough data for medication pattern analysis."
    
    try:
        # Extract medication data
        med_data = []
        for item in data.get("medications", []):
            if "date" in item and "name" in item:
                med_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "name": item["name"],
                    "quantity": item.get("quantity", 1)
                })
        
        # Get mood/symptom data
        mood_data = []
        for item in data["symptoms"]:
            if "date" in item and "type" in item and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "type": item["type"],
                    "severity": item["severity"]
                })
        
        if len(med_data) < 5 or len(mood_data) < 5:
            return "Not enough data points for medication analysis."
        
        # Create DataFrames
        med_df = pd.DataFrame(med_data)
        mood_df = pd.DataFrame(mood_data)
        
        # Group by day to create transactions
        med_df["day"] = med_df["date"].dt.date
        mood_df["day"] = mood_df["date"].dt.date
        
        # Create transactions dataset
        days = sorted(set(list(med_df["day"]) + list(mood_df["day"])))
        transactions = []
        
        for day in days:
            day_meds = med_df[med_df["day"] == day]["name"].unique().tolist()
            
            # For mood, get the average severity for that day
            day_mood_df = mood_df[(mood_df["day"] == day) & (mood_df["type"] == mood_field)]
            
            if not day_mood_df.empty:
                avg_severity = day_mood_df["severity"].mean()
                mood_level = f"{mood_field}_Level_{round(avg_severity)}"
                transaction = day_meds + [mood_level]
                transactions.append(transaction)
        
        if len(transactions) < 5:
            return "Not enough daily data for pattern analysis."
        
        # Apply Apriori algorithm for frequent itemsets
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        
        if frequent_itemsets.empty:
            return "No significant patterns found with current support threshold."
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        if rules.empty:
            return "No strong association rules found."
        
        # Filter rules related to mood levels
        mood_rules = []
        for _, rule in rules.iterrows():
            antecedents = list(rule["antecedents"])
            # Checking if antecedents or consequents contain mood level ratings
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
            
        # Sort by lift (importance)
        mood_rules.sort(key=lambda x: x["lift"], reverse=True)
        
        return mood_rules[:5]  # Return top 5 rules
    except Exception as e:
        return f"Error in medication pattern analysis: {str(e)}"

# New function: Analyze activity patterns
def analyze_activity_patterns(data, mood_field):
    if not data or "activities" not in data or "symptoms" not in data:
        return "Not enough data for activity pattern analysis."
    
    try:
        # Extract activity data
        activity_data = []
        for item in data.get("activities", []):
            if "date" in item and "activityName" in item and "duration" in item and "intensity" in item:
                activity_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "name": item["activityName"],
                    "duration": item.get("duration", 0),
                    "intensity": item.get("intensity", "Low"),
                    "notes": item.get("notes", "")
                })
        
        # Get mood/symptom data
        mood_data = []
        for item in data["symptoms"]:
            if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                mood_data.append({
                    "date": pd.to_datetime(item["date"]),
                    "severity": item.get("severity", 0)
                })
        
        if len(activity_data) < 5 or len(mood_data) < 5:
            return "Not enough data points for activity analysis."
        
        # Create DataFrames
        activity_df = pd.DataFrame(activity_data)
        mood_df = pd.DataFrame(mood_data)
        
        # Convert intensity to numeric
        intensity_map = {"Low": 1, "Moderate": 2, "High": 3}
        activity_df["intensity_score"] = activity_df["intensity"].map(lambda x: intensity_map.get(x, 1))
        
        # Calculate activity score (duration * intensity)
        activity_df["activity_score"] = activity_df["duration"] * activity_df["intensity_score"]
        
        # Match activities with mood (within 6 hours)
        matched_data = []
        
        for _, act_row in activity_df.iterrows():
            act_date = act_row["date"]
            
            # Find mood measurements after activity (within 6 hours)
            relevant_moods = mood_df[(mood_df["date"] >= act_date) & 
                                    (mood_df["date"] <= act_date + pd.Timedelta(hours=6))]
            
            if not relevant_moods.empty:
                # Take the average mood if multiple entries exist
                avg_mood = relevant_moods["severity"].mean()
                
                matched_data.append({
                    "date": act_date,
                    "activity_name": act_row["name"],
                    "duration": act_row["duration"],
                    "intensity": act_row["intensity"],
                    "activity_score": act_row["activity_score"],
                    "mood_after": avg_mood
                })
        
        if len(matched_data) < 3:
            return "Not enough matched activity-mood data for analysis."
        
        matched_df = pd.DataFrame(matched_data)
        
        # Group by activity name
        activity_analysis = []
        
        for activity_name, group in matched_df.groupby("activity_name"):
            if len(group) >= 2:  # At least 2 instances
                avg_duration = group["duration"].mean()
                avg_score = group["activity_score"].mean()
                avg_mood = group["mood_after"].mean()
                
                # Correlation between activity_score and mood (if enough data points)
                correlation = None
                if len(group) >= 3:
                    if group["activity_score"].std() > 0 and group["mood_after"].std() > 0:
                        correlation, _ = pearsonr(group["activity_score"], group["mood_after"])
                
                activity_analysis.append({
                    "activity_name": activity_name,
                    "count": len(group),
                    "avg_duration": round(avg_duration, 1),
                    "avg_mood_after": round(avg_mood, 2),
                    "correlation": round(correlation, 3) if correlation is not None else None
                })
        
        # Sort by mood impact (highest first)
        activity_analysis.sort(key=lambda x: x["avg_mood_after"], reverse=True)
        
        return activity_analysis
    except Exception as e:
        return f"Error in activity pattern analysis: {str(e)}"

# New function: Create comprehensive pattern analysis dashboard
def create_pattern_dashboard(data):
    if not data:
        return "No data available for analysis."
    
    try:
        # Define mood fields to analyze
        mood_fields = ["Parkinson's State", "Physical State", "My Mood"]
        dashboard_data = {}
        
        # For each mood/state field
        for mood_field in mood_fields:
            field_results = {}
            
            # Analyze nutrition patterns
            nutrition_clusters = analyze_nutrition_clusters(data, mood_field)
            field_results["nutrition_patterns"] = nutrition_clusters
            
            # Analyze medication patterns
            medication_patterns = analyze_medication_patterns(data, mood_field)
            field_results["medication_patterns"] = medication_patterns
            
            # Analyze activity patterns
            activity_patterns = analyze_activity_patterns(data, mood_field)
            field_results["activity_patterns"] = activity_patterns
            
            # Extract time patterns for each data type
            for field_type in ["nutritions", "medications", "activities"]:
                time_patterns = extract_time_patterns(data, field_type, mood_field)
                field_results[f"{field_type}_time_patterns"] = time_patterns
            
            dashboard_data[mood_field] = field_results
        
        return dashboard_data
    except Exception as e:
        return f"Error creating dashboard: {str(e)}"

# Function to summarize insights in text format
def summarize_insights(dashboard_data):
    if not isinstance(dashboard_data, dict):
        return "No data available for insights."
    
    insights = []
    
    try:
        # Process each mood field
        for mood_field, field_data in dashboard_data.items():
            insights.append(f"## Insights for {mood_field}:")
            
            # 1. Nutrition insights
            nutrition_patterns = field_data.get("nutrition_patterns", [])
            if isinstance(nutrition_patterns, list) and len(nutrition_patterns) > 0:
                insights.append("\n### Nutrition Patterns:")
                
                # Sort clusters by mood effect
                best_clusters = sorted(nutrition_patterns, key=lambda x: x.get("avg_mood", 0))
                
                if len(best_clusters) > 0:
                    best_cluster = best_clusters[0]  # Cluster with best mood
                    worst_cluster = best_clusters[-1] if len(best_clusters) > 1 else None  # Cluster with worst mood
                    
                    if best_cluster:
                        foods = ", ".join([f"{food[0]} ({food[1]} times)" for food in best_cluster.get("common_foods", [])[:2]])
                        insights.append(f"- **Beneficial Nutrition Profile**: Foods high in "
                                      f"Protein: {best_cluster.get('nutrition_profile', {}).get('proteins', 0)}g, "
                                      f"Fiber: {best_cluster.get('nutrition_profile', {}).get('fiber', 0)}g "
                                      f"(like {foods}) associated with better {mood_field} ratings.")
                    
                    if worst_cluster:
                        foods = ", ".join([f"{food[0]} ({food[1]} times)" for food in worst_cluster.get("common_foods", [])[:2]])
                        insights.append(f"- **Less Beneficial Nutrition**: Foods with "
                                      f"Fat: {worst_cluster.get('nutrition_profile', {}).get('fats', 0)}g, "
                                      f"Carbs: {worst_cluster.get('nutrition_profile', {}).get('carbs', 0)}g "
                                      f"(like {foods}) associated with lower {mood_field} ratings.")
            
            # 2. Medication insights
            medication_patterns = field_data.get("medication_patterns", [])
            if isinstance(medication_patterns, list) and len(medication_patterns) > 0:
                insights.append("\n### Medication Patterns:")
                
                for idx, rule in enumerate(medication_patterns[:3]):
                    antecedents = list(rule.get("antecedents", []))
                    consequents = list(rule.get("consequents", []))
                    
                    # Extract medications and mood levels
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
                        insights.append(f"- **Medication Association**: {meds_str} associated with {mood_field} {mood_level} "
                                      f"(Confidence: {rule.get('confidence', 0):.2f}, Lift: {rule.get('lift', 0):.2f})")
            
            # 3. Activity insights
            activity_patterns = field_data.get("activity_patterns", [])
            if isinstance(activity_patterns, list) and len(activity_patterns) > 0:
                insights.append("\n### Activity Patterns:")
                
                for idx, activity in enumerate(activity_patterns[:3]):
                    insights.append(f"- **{activity.get('activity_name', 'Activity')}**: "
                                  f"Average duration {activity.get('avg_duration', 0)} minutes, "
                                  f"associated with {mood_field} rating of {activity.get('avg_mood_after', 0):.1f}/5 afterward.")
                    
                    if activity.get('correlation') is not None:
                        corr = activity.get('correlation')
                        if abs(corr) > 0.3:
                            direction = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs(corr) > 0.7 else "moderate"
                            insights.append(f"  - {strength.capitalize()} {direction} correlation ({corr:.2f}) between activity intensity and {mood_field}")
            
            # 4. Time pattern insights
            for field_type in ["nutritions", "medications", "activities"]:
                time_patterns = field_data.get(f"{field_type}_time_patterns", {})
                
                if time_patterns and isinstance(time_patterns, dict):
                    morning_mood = time_patterns.get("morning_mood")
                    afternoon_mood = time_patterns.get("afternoon_mood")
                    evening_mood = time_patterns.get("evening_mood")
                    
                    if morning_mood is not None and afternoon_mood is not None and evening_mood is not None:
                        best_time = "morning" if morning_mood >= max(afternoon_mood, evening_mood) else "afternoon" if afternoon_mood >= max(morning_mood, evening_mood) else "evening"
                        
                        insights.append(f"\n- **{field_type.capitalize()} Timing**: Best {mood_field} ratings observed in the {best_time} "
                                      f"(M: {morning_mood:.1f}, A: {afternoon_mood:.1f}, E: {evening_mood:.1f})")
        
        # Join all insights with line breaks
        return "\n".join(insights)
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Create visualizations for patterns
def create_pattern_visualizations(data):
    try:
        # Create temporary directory for visualizations
        os.makedirs("temp_visuals", exist_ok=True)
        visual_paths = []
        
        # Select a mood field for visualization
        mood_field = "Parkinson's State"  # Default, can be changed
        
        # 1. Medication timing vs mood visualization
        if "medications" in data and "symptoms" in data:
            med_data = []
            for item in data.get("medications", []):
                if "date" in item and "name" in item:
                    med_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "name": item["name"],
                        "hour": pd.to_datetime(item["date"]).hour
                    })
            
            mood_data = []
            for item in data["symptoms"]:
                if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                    mood_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "severity": item.get("severity", 0),
                        "hour": pd.to_datetime(item["date"]).hour
                    })
            
            if med_data and mood_data:
                med_df = pd.DataFrame(med_data)
                mood_df = pd.DataFrame(mood_data)
                
                # Group by hour
                med_by_hour = med_df.groupby("hour").size().reset_index(name="count")
                mood_by_hour = mood_df.groupby("hour")["severity"].mean().reset_index()
                
                # Create dual-axis visualization
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.set_xlabel("Hour of Day")
                ax1.set_ylabel("Number of Medications", color="tab:blue")
                ax1.bar(med_by_hour["hour"], med_by_hour["count"], color="tab:blue", alpha=0.7)
                ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.set_xticks(range(0, 24, 2))
                
                ax2 = ax1.twinx()
                ax2.set_ylabel(f"{mood_field} Rating", color="tab:red")
                ax2.plot(mood_by_hour["hour"], mood_by_hour["severity"], "o-", color="tab:red")
                ax2.tick_params(axis="y", labelcolor="tab:red")
                ax2.set_ylim(0, 5.5)
                
                plt.title(f"Medication Timing vs {mood_field} Ratings")
                plt.tight_layout()
                
                med_vis_path = os.path.join("temp_visuals", "medication_timing.png")
                plt.savefig(med_vis_path)
                plt.close()
                visual_paths.append(med_vis_path)
        
        # 2. Nutrition composition vs mood
        if "nutritions" in data and "symptoms" in data:
            nutrition_data = []
            for item in data.get("nutritions", []):
                if "nutritionalValues" in item and "date" in item:
                    nutrition_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "proteins": item["nutritionalValues"].get("proteins", 0),
                        "fats": item["nutritionalValues"].get("fats", 0),
                        "carbs": item["nutritionalValues"].get("carbohydrates", 0),
                        "fiber": item["nutritionalValues"].get("dietaryFiber", 0)
                    })
            
            mood_data = []
            for item in data["symptoms"]:
                if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                    mood_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "severity": item.get("severity", 0)
                    })
            
            if nutrition_data and mood_data:
                nutr_df = pd.DataFrame(nutrition_data)
                mood_df = pd.DataFrame(mood_data)
                
                # Join nutrition with closest mood (within 3 hours)
                merged_data = []
                for _, nutr_row in nutr_df.iterrows():
                    nutr_date = nutr_row["date"]
                    closest_mood = None
                    min_diff = float('inf')
                    
                    for _, mood_row in mood_df.iterrows():
                        mood_date = mood_row["date"]
                        diff = abs((nutr_date - mood_date).total_seconds() / 3600)
                        
                        if diff < min_diff and diff <= 3:
                            min_diff = diff
                            closest_mood = mood_row["severity"]
                    
                    if closest_mood is not None:
                        merged_data.append({
                            "date": nutr_date,
                            "proteins": nutr_row["proteins"],
                            "fats": nutr_row["fats"],
                            "carbs": nutr_row["carbs"],
                            "fiber": nutr_row["fiber"],
                            "mood": closest_mood
                        })
                
                if len(merged_data) >= 5:
                    merged_df = pd.DataFrame(merged_data)
                    
                    # Create PCA visualization
                    X = merged_df[["proteins", "fats", "carbs", "fiber"]]
                    y = merged_df["mood"]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Create visualization
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=100, alpha=0.7)
                    plt.colorbar(scatter, label=f"{mood_field} Rating")
                    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                    plt.title(f"Nutrition Composition vs {mood_field} (PCA)")
                    
                    # Add feature vectors
                    feature_names = ["Proteins", "Fats", "Carbs", "Fiber"]
                    for i, (name, vec) in enumerate(zip(feature_names, pca.components_.T)):
                        plt.arrow(0, 0, vec[0]*5, vec[1]*5, color='r', alpha=0.5)
                        plt.text(vec[0]*5.2, vec[1]*5.2, name, color='r')
                    
                    plt.tight_layout()
                    nutr_vis_path = os.path.join("temp_visuals", "nutrition_pca.png")
                    plt.savefig(nutr_vis_path)
                    plt.close()
                    visual_paths.append(nutr_vis_path)
                    
                    # Create correlation heatmap
                    corr_matrix = merged_df.corr()
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                    plt.title(f"Correlation Between Nutrients and {mood_field}")
                    plt.tight_layout()
                    
                    corr_vis_path = os.path.join("temp_visuals", "nutrition_correlation.png")
                    plt.savefig(corr_vis_path)
                    plt.close()
                    visual_paths.append(corr_vis_path)
        
        # 3. Activity intensity vs mood
        if "activities" in data and "symptoms" in data:
            activity_data = []
            for item in data.get("activities", []):
                if "date" in item and "intensity" in item and "duration" in item:
                    intensity_value = {"Low": 1, "Moderate": 2, "High": 3}.get(item["intensity"], 1)
                    activity_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "intensity": intensity_value,
                        "duration": item.get("duration", 0),
                        "score": intensity_value * item.get("duration", 0)
                    })
            
            mood_data = []
            for item in data["symptoms"]:
                if "date" in item and "type" in item and item.get("type") == mood_field and "severity" in item:
                    mood_data.append({
                        "date": pd.to_datetime(item["date"]),
                        "severity": item.get("severity", 0)
                    })
            
            if activity_data and mood_data:
                act_df = pd.DataFrame(activity_data)
                mood_df = pd.DataFrame(mood_data)
                
                # Find mood after activity (within 6 hours)
                matched_data = []
                for _, act_row in act_df.iterrows():
                    act_date = act_row["date"]
                    relevant_moods = mood_df[(mood_df["date"] >= act_date) & 
                                          (mood_df["date"] <= act_date + pd.Timedelta(hours=6))]
                    
                    if not relevant_moods.empty:
                        avg_mood = relevant_moods["severity"].mean()
                        matched_data.append({
                            "activity_score": act_row["score"],
                            "duration": act_row["duration"],
                            "intensity": act_row["intensity"],
                            "mood_after": avg_mood
                        })
                
                if len(matched_data) >= 5:
                    matched_df = pd.DataFrame(matched_data)
                    
                    # Create scatter plot
                    plt.figure(figsize=(10, 6))
                    plt.scatter(matched_df["activity_score"], matched_df["mood_after"], 
                             c=matched_df["intensity"], cmap="plasma", s=80, alpha=0.7)
                    
                    # Add regression line
                    x = matched_df["activity_score"]
                    y = matched_df["mood_after"]
                    m, b = np.polyfit(x, y, 1)
                    plt.plot(x, m*x + b, color="red", linestyle="--")
                    
                    # Calculate correlation
                    corr, _ = pearsonr(matched_df["activity_score"], matched_df["mood_after"])
                    
                    plt.colorbar(label="Activity Intensity")
                    plt.xlabel("Activity Score (Duration × Intensity)")
                    plt.ylabel(f"{mood_field} Rating After Activity")
                    plt.title(f"Activity Score vs {mood_field} Rating (Correlation: {corr:.2f})")
                    plt.tight_layout()
                    
                    act_vis_path = os.path.join("temp_visuals", "activity_mood.png")
                    plt.savefig(act_vis_path)
                    plt.close()
                    visual_paths.append(act_vis_path)
        
        return visual_paths
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return []

# Update the Gradio interface
def create_analysis_interface():
    # Add new components for pattern analysis
    with gr.Blocks(title="Parkinson's Data Analysis") as app:
        gr.Markdown("# Parkinson's Health Data Analytics")
        gr.Markdown("Upload your data file to analyze patterns and gain insights")
        
        with gr.Row():
            file_input = gr.File(label="Upload JSON File")
            process_button = gr.Button("Process Data & Analyze Patterns")
        
        output_text = gr.Textbox(label="Processing Status")
        processed_file = gr.File(label="Download Processed File")
        
        # Add tabs for different analyses
        with gr.Tabs():
            with gr.TabItem("Insights & Patterns"):
                pattern_field = gr.Dropdown(
                    ["Parkinson's State", "Physical State", "My Mood"], 
                    label="Select Health Metric to Analyze", 
                    value="Parkinson's State"
                )
                analyze_button = gr.Button("Generate Insights")
                insight_output = gr.Markdown(label="Key Insights")
            
            with gr.TabItem("Nutrition Analysis"):
                gr.Markdown("### Nutrition Impact Analysis")
                nutrition_field = gr.Dropdown(
                    ["Parkinson's State", "Physical State", "My Mood"], 
                    label="Select Health Metric", 
                    value="Parkinson's State"
                )
                nutrition_button = gr.Button("Analyze Nutrition Patterns")
                nutrition_output = gr.JSON(label="Nutrition Clusters")
            
            with gr.TabItem("Medication Analysis"):
                gr.Markdown("### Medication Impact Analysis")
                medication_field = gr.Dropdown(
                    ["Parkinson's State", "Physical State", "My Mood"], 
                    label="Select Health Metric", 
                    value="Parkinson's State"
                )
                medication_button = gr.Button("Analyze Medication Patterns")
                medication_output = gr.JSON(label="Medication Associations")
            
            with gr.TabItem("Activity Analysis"):
                gr.Markdown("### Activity Impact Analysis")
                activity_field = gr.Dropdown(
                    ["Parkinson's State", "Physical State", "My Mood"], 
                    label="Select Health Metric", 
                    value="Parkinson's State"
                )
                activity_button = gr.Button("Analyze Activity Patterns")
                activity_output = gr.JSON(label="Activity Patterns")
            
            with gr.TabItem("Visualizations"):
                gr.Markdown("### Data Visualizations")
                visual_button = gr.Button("Generate Visualizations")
                with gr.Row():
                    visual_gallery = gr.Gallery(label="Pattern Visualizations")

        
        # Process file and data analysis
        process_button.click(
            fn=upload_and_process, 
            inputs=[file_input], 
            outputs=[processed_file, output_text]
        )
        
        # Pattern analysis
        analyze_button.click(
            fn=lambda field: summarize_insights(create_pattern_dashboard(translated_data_global))[field] if translated_data_global else "Please upload and process data first.",
            inputs=[pattern_field],
            outputs=[insight_output]
        )
        
        # Nutrition analysis
        nutrition_button.click(
            fn=lambda field: analyze_nutrition_clusters(translated_data_global, field) if translated_data_global else "Please upload and process data first.",
            inputs=[nutrition_field],
            outputs=[nutrition_output]
        )
        
        # Medication analysis
        medication_button.click(
            fn=lambda field: analyze_medication_patterns(translated_data_global, field) if translated_data_global else "Please upload and process data first.",
            inputs=[medication_field],
            outputs=[medication_output]
        )
        
        # Activity analysis
        activity_button.click(
            fn=lambda field: analyze_activity_patterns(translated_data_global, field) if translated_data_global else "Please upload and process data first.",
            inputs=[activity_field],
            outputs=[activity_output]
        )
        
        # Visualizations
        visual_button.click(
            fn=lambda: create_pattern_visualizations(translated_data_global) if translated_data_global else [],
            inputs=[],
            outputs=[visual_gallery]
        )
    
    return app

# Final app creation
app = create_analysis_interface()

# Launch the application
if __name__ == "__main__":
    app.launch()
