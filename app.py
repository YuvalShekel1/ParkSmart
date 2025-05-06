import gradio as gr
import pandas as pd
import numpy as np
import json
import os
import tempfile
from collections import Counter
import datetime

# גלובלים
translated_data_global = {}
processed_file_path = ""

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

# --- ניתוחים ודפוסים ---

def generate_activity_insights(activity_df, mood_df):
    insights = "🏃 תובנות פעילות:\n"

    if activity_df.empty or mood_df.empty:
        return insights + "אין מספיק נתונים לניתוח הקשר בין פעילויות ומצב רוח.\n"

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
                activity_type = activity_item.get("activityType", "לא ידוע")
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
        return insights + "אין מספיק נתונים בעלי תזמון קרוב.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• השפעת סוג פעילות על מצב הרוח:\n"
    activity_types = analysis_df["activity_type"].unique()
    for activity in activity_types:
        act_mood = analysis_df[analysis_df["activity_type"] == activity]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = act_mood - overall_mood
        if not np.isnan(act_mood):
            count = len(analysis_df[analysis_df["activity_type"] == activity])
            direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
            if abs(diff) >= 0.5:
                insights += f"  - {activity} ({count} פעמים): מצב הרוח נוטה להיות {abs(round(diff, 1))} נקודות {direction} מהממוצע.\n"
            else:
                insights += f"  - {activity} ({count} פעמים): מצב הרוח דומה לממוצע.\n"

    # ניתוח נוסף - אורך פעילות
    if "duration" in analysis_df.columns and analysis_df["duration"].sum() > 0:
        # קטגוריות אורך הפעילות
        analysis_df["duration_category"] = pd.cut(
            analysis_df["duration"], 
            bins=[0, 15, 30, 60, float('inf')],
            labels=["קצר (0-15 דקות)", "בינוני (15-30 דקות)", "ארוך (30-60 דקות)", "ממושך (60+ דקות)"]
        )
        
        insights += "\n• השפעת אורך הפעילות על מצב הרוח:\n"
        duration_groups = analysis_df.groupby("duration_category")["mood_value"].mean()
        
        for category, mean_mood in duration_groups.items():
            if pd.notnull(mean_mood):
                diff = mean_mood - overall_mood
                count = len(analysis_df[analysis_df["duration_category"] == category])
                direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
                if abs(diff) >= 0.3 and count >= 2:
                    insights += f"  - פעילויות {category} ({count} פעמים): מצב הרוח בדרך כלל {abs(round(diff, 1))} נקודות {direction}.\n"

    # ניתוח נוסף - השפעת עצימות
    if "intensity" in analysis_df.columns and analysis_df["intensity"].sum() > 0:
        # קטגוריות עצימות
        analysis_df["intensity_category"] = pd.cut(
            analysis_df["intensity"], 
            bins=[0, 3, 7, 10],
            labels=["עצימות נמוכה", "עצימות בינונית", "עצימות גבוהה"]
        )
        
        insights += "\n• השפעת עצימות הפעילות על מצב הרוח:\n"
        intensity_groups = analysis_df.groupby("intensity_category")["mood_value"].mean()
        
        for category, mean_mood in intensity_groups.items():
            if pd.notnull(mean_mood):
                diff = mean_mood - overall_mood
                count = len(analysis_df[analysis_df["intensity_category"] == category])
                direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
                if abs(diff) >= 0.3 and count >= 2:
                    insights += f"  - פעילויות ב{category} ({count} פעמים): מצב הרוח בדרך כלל {abs(round(diff, 1))} נקודות {direction}.\n"

    return insights

def generate_medication_insights(medication_df, mood_df):
    insights = "💊 תובנות תרופות:\n"

    if medication_df.empty or mood_df.empty:
        return insights + "אין מספיק נתונים לניתוח תרופות.\n"

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
                med_name = med_item.get("medicationName", "לא ידוע")
                dosage = med_item.get("dosage", 0)
                combined_data.append({
                    "mood_value": mood_value,
                    "medication": med_name,
                    "dosage": dosage,
                    "time_diff": time_diff
                })

    if not combined_data or len(combined_data) < 2:
        return insights + "אין מספיק נתונים בעלי תזמון קרוב.\n"

    analysis_df = pd.DataFrame(combined_data)
    insights += "• השפעת תרופות על מצב הרוח:\n"
    medications = analysis_df["medication"].unique()
    for med in medications:
        med_mood = analysis_df[analysis_df["medication"] == med]["mood_value"].mean()
        overall_mood = mood_df["value"].mean()
        diff = med_mood - overall_mood
        if not np.isnan(med_mood):
            count = len(analysis_df[analysis_df["medication"] == med])
            direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
            if abs(diff) >= 0.3:
                insights += f"  - {med} ({count} פעמים): מצב הרוח נוטה להיות {abs(round(diff, 1))} נקודות {direction}.\n"
            else:
                insights += f"  - {med} ({count} פעמים): מצב הרוח דומה לממוצע.\n"

    # ניתוח עיתוי לקיחת תרופות
    if "time_diff" in analysis_df.columns:
        insights += "\n• השפעת העיתוי של לקיחת תרופות:\n"
        # קטגוריות עיתוי
        analysis_df["timing_category"] = pd.cut(
            analysis_df["time_diff"], 
            bins=[0, 1, 3, 6],
            labels=["תוך שעה", "1-3 שעות", "3-6 שעות"]
        )
        
        timing_groups = analysis_df.groupby("timing_category")["mood_value"].mean()
        
        for category, mean_mood in timing_groups.items():
            if pd.notnull(mean_mood):
                diff = mean_mood - overall_mood
                count = len(analysis_df[analysis_df["timing_category"] == category])
                direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
                if abs(diff) >= 0.3 and count >= 2:
                    insights += f"  - לקיחת תרופות {category} מזמן הדיווח ({count} פעמים): מצב הרוח בדרך כלל {abs(round(diff, 1))} נקודות {direction}.\n"

    return insights

def generate_symptom_insights(symptom_df, mood_df):
    insights = "🩺 תובנות סימפטומים:\n"

    if symptom_df.empty or mood_df.empty:
        return insights + "אין מספיק נתונים לניתוח סימפטומים.\n"

    symptom_fields = set()
    for _, row in symptom_df.iterrows():
        item = row["item"]
        for key in item.keys():
            if key not in ["date", "notes", "id", "Parkinson's State", "My Mood", "Physical State"]:
                symptom_fields.add(key)
    symptom_fields = list(symptom_fields)

    if not symptom_fields:
        return insights + "לא זוהו שדות סימפטומים ספציפיים.\n"

    insights += "• השפעת סימפטומים על מצב הרוח:\n"
    date_to_mood = {row["date"].date(): row["value"] for _, row in mood_df.iterrows()}

    for symptom in symptom_fields:
        symptom_present_moods = []
        symptom_absent_moods = []
        for _, row in symptom_df.iterrows():
            date = row["date"].date()
            item = row["item"]
            if date in date_to_mood:
                mood_value = date_to_mood[date]
                if symptom in item and item[symptom]:
                    symptom_present_moods.append(mood_value)
                else:
                    symptom_absent_moods.append(mood_value)
        if symptom_present_moods and symptom_absent_moods:
            present_avg = np.mean(symptom_present_moods)
            absent_avg = np.mean(symptom_absent_moods)
            diff = present_avg - absent_avg
            direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
            if abs(diff) >= 0.3:
                insights += f"  - {symptom}: מצב הרוח {direction} ב-{round(abs(diff),1)} נקודות כאשר הסימפטום נוכח.\n"
            else:
                insights += f"  - {symptom}: אין השפעה חזקה על מצב הרוח.\n"

    # ניתוח צירופי סימפטומים
    if len(symptom_fields) > 1:
        insights += "\n• השפעת צירופי סימפטומים:\n"
        symptom_combos = []
        
        for _, row in symptom_df.iterrows():
            date = row["date"].date()
            item = row["item"]
            if date in date_to_mood:
                present_symptoms = []
                for symptom in symptom_fields:
                    if symptom in item and item[symptom]:
                        present_symptoms.append(symptom)
                
                if len(present_symptoms) > 1:
                    combo = " + ".join(sorted(present_symptoms))
                    symptom_combos.append({
                        "combo": combo,
                        "mood": date_to_mood[date]
                    })
        
        combo_df = pd.DataFrame(symptom_combos)
        if not combo_df.empty:
            combo_counts = Counter(combo_df["combo"])
            for combo, count in combo_counts.items():
                if count >= 2:
                    combo_mood = combo_df[combo_df["combo"] == combo]["mood"].mean()
                    overall_mood = mood_df["value"].mean()
                    diff = combo_mood - overall_mood
                    direction = "גבוה יותר" if diff > 0 else "נמוך יותר"
                    if abs(diff) >= 0.5:
                        insights += f"  - צירוף {combo} ({count} פעמים): מצב הרוח {direction} ב-{round(abs(diff),1)} נקודות.\n"
        else:
            insights += "  אין מספיק נתונים על צירופי סימפטומים.\n"

    return insights

# --- פונקציות נוספות: ערכים תזונתיים ותובנות נוספות ---

def get_nutritional_info(data):
    if not data or "meals" not in data:
        return "אין נתונים תזונתיים זמינים."
    
    meals = data.get("meals", [])
    if not meals:
        return "אין נתונים על ארוחות."
    
    nutritional_info = "📊 מידע תזונתי:\n\n"
    
    # איסוף נתונים תזונתיים
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    food_counts = Counter()
    
    for meal in meals:
        if "foods" in meal:
            for food in meal["foods"]:
                food_name = food.get("foodName", "לא ידוע")
                calories = food.get("calories", 0)
                protein = food.get("protein", 0)
                carbs = food.get("carbohydrates", 0)
                fat = food.get("fat", 0)
                
                total_calories += calories
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
                food_counts[food_name] += 1
    
    # חישוב ממוצעים יומיים
    days = len(set([meal.get("date").split("T")[0] for meal in meals if "date" in meal]))
    if days == 0:
        days = 1
    
    avg_calories = total_calories / days
    avg_protein = total_protein / days
    avg_carbs = total_carbs / days
    avg_fat = total_fat / days
    
    nutritional_info += f"• ממוצע יומי:\n"
    nutritional_info += f"  - קלוריות: {round(avg_calories)} קק\"ל\n"
    nutritional_info += f"  - חלבון: {round(avg_protein, 1)} גרם\n"
    nutritional_info += f"  - פחמימות: {round(avg_carbs, 1)} גרם\n"
    nutritional_info += f"  - שומן: {round(avg_fat, 1)} גרם\n\n"
    
    # המזונות הנפוצים ביותר
    nutritional_info += f"• מזונות נפוצים:\n"
    for food, count in food_counts.most_common(5):
        nutritional_info += f"  - {food}: {count} פעמים\n"
    
    return nutritional_info

def generate_comprehensive_insights(data, mood_field):
    if not data:
        return "נא להעלות ולעבד נתונים תחילה."
    
    activity_df, mood_df = prepare_activity_and_mood_data(data, mood_field)
    medication_df, mood_df_med = prepare_medication_and_mood_data(data, mood_field)
    symptom_df, mood_df_symp = prepare_symptom_and_mood_data(data, mood_field)
    
    # איסוף כל התובנות
    activity_insights = generate_activity_insights(activity_df, mood_df)
    medication_insights = generate_medication_insights(medication_df, mood_df_med)
    symptom_insights = generate_symptom_insights(symptom_df, mood_df_symp)
    
    # תובנות על מגמות זמן
    time_insights = "📅 תובנות מגמות זמן:\n"
    
    if not mood_df.empty:
        mood_df = mood_df.sort_values("date")
        mood_df["day_of_week"] = mood_df["date"].dt.day_name()
        
        # מגמה לאורך זמן
        if len(mood_df) > 5:
            first_week = mood_df.iloc[:len(mood_df)//2]["value"].mean()
            second_week = mood_df.iloc[len(mood_df)//2:]["value"].mean()
            diff = second_week - first_week
            if abs(diff) >= 0.3:
                direction = "עלייה" if diff > 0 else "ירידה"
                time_insights += f"• נראית {direction} כללית במצב הרוח של {abs(round(diff, 1))} נקודות במחצית השנייה של התקופה.\n"
        
        # ניתוח לפי יום בשבוע
        dow_mood = mood_df.groupby("day_of_week")["value"].mean()
        if len(dow_mood) > 1:
            overall_mean = mood_df["value"].mean()
            time_insights += "• מצב רוח לפי יום בשבוע:\n"
            
            # מיפוי שמות ימים מאנגלית לעברית
            day_names = {
                "Monday": "יום שני",
                "Tuesday": "יום שלישי",
                "Wednesday": "יום רביעי",
                "Thursday": "יום חמישי",
                "Friday": "יום שישי",
                "Saturday": "יום שבת",
                "Sunday": "יום ראשון"
            }
            
            for day, mood_val in dow_mood.items():
                hebrew_day = day_names.get(day, day)
                day_count = len(mood_df[mood_df["day_of_week"] == day])
                diff = mood_val - overall_mean
                if abs(diff) >= 0.3 and day_count >= 2:
                    direction = "טוב יותר" if diff > 0 else "פחות טוב"
                    time_insights += f"  - {hebrew_day}: מצב רוח {direction} ב-{abs(round(diff, 1))} נקודות ({day_count} תצפיות).\n"
    else:
        time_insights += "אין מספיק נתונים למגמות זמן.\n"
    
    # חיבור כל התובנות
    all_insights = "📊 תובנות מקיפות 📊\n\n"
    all_insights += activity_insights + "\n\n"
    all_insights += medication_insights + "\n\n"
    all_insights += symptom_insights + "\n\n"
    all_insights += time_insights
    
    return all_insights

# --- חיבור לגרידיו ---

def upload_json(file_obj):
    global translated_data_global, processed_file_path
    if file_obj is None:
        return None, "❌ לא הועלה קובץ."
    try:
        # טיפול בסוג הקובץ הנכון
        if hasattr(file_obj, 'read'):
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            data = json.loads(content)
        else:
            # אם כבר מקבלים מחרוזת או תוכן, ננסה לפענח ישירות
            content = file_obj
            data = json.loads(content)
            
        translated_data_global = data
        
        # שמירת הקובץ המעובד
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        with open(temp_path.name, "w", encoding="utf-8") as f:
            json.dump(translated_data_global, f, ensure_ascii=False, indent=2)
        processed_file_path = temp_path.name
        
        return processed_file_path, "✅ הקובץ הועלה ועובד בהצלחה."
    except Exception as e:
        return None, f"❌ שגיאה: {str(e)}"

def activity_analysis_summary(mood_field):
    if not translated_data_global:
        return "נא להעלות ולעבד נתונים תחילה."
    activity_df, mood_df = prepare_activity_and_mood_data(translated_data_global, mood_field)
    return generate_activity_insights(activity_df, mood_df)

def medication_analysis_summary(mood_field):
    if not translated_data_global:
        return "נא להעלות ולעבד נתונים תחילה."
    medication_df, mood_df = prepare_medication_and_mood_data(translated_data_global, mood_field)
    return generate_medication_insights(medication_df, mood_df)

def symptom_analysis_summary(mood_field):
    if not translated_data_global:
        return "נא להעלות ולעבד נתונים תחילה."
    symptom_df, mood_df = prepare_symptom_and_mood_data(translated_data_global, mood_field)
    return generate_symptom_insights(symptom_df, mood_df)

def nutritional_analysis_summary():
    if not translated_data_global:
        return "נא להעלות ולעבד נתונים תחילה."
    return get_nutritional_info(translated_data_global)

def comprehensive_analysis_summary(mood_field):
    if not translated_data_global:
        return "נא להעלות ולעבד נתונים תחילה."
    return generate_comprehensive_insights(translated_data_global, mood_field)

def download_processed_file():
    if processed_file_path:
        return processed_file_path
    return None

# --- יצירת האפליקציה ---

with gr.Blocks(title="ניתוח דפוסי בריאות וסימפטומים") as app:
    gr.Markdown("# 📈 ניתוח דפוסי בריאות וסימפטומים")

    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(label="העלאת קובץ JSON")
        with gr.Column(scale=2):
            upload_button = gr.Button("העלאה ועיבוד", variant="primary")
    
    output_text = gr.Textbox(label="סטטוס", interactive=False)
    processed_file = gr.File(label="הורדת קובץ מעובד", interactive=False)

    mood_selector = gr.Dropdown(
        ["Parkinson's State", "Physical State", "My Mood"],
        label="בחירת שדה מצב רוח",
        value="My Mood"
    )

    with gr.Tabs():
        with gr.TabItem("🏃 ניתוח פעילות"):
            activity_button = gr.Button("ניתוח דפוסי פעילות")
            activity_output = gr.Markdown(label="תובנות פעילות")
        
        with gr.TabItem("💊 ניתוח תרופות"):
            medication_button = gr.Button("ניתוח דפוסי תרופות")
            medication_output = gr.Markdown(label="תובנות תרופות")

        with gr.TabItem("🩺 ניתוח סימפטומים"):
            symptom_button = gr.Button("ניתוח דפוסי סימפטומים")
            symptom_output = gr.Markdown(label="תובנות סימפטומים")
            
        with gr.TabItem("🍎 ניתוח תזונה"):
            nutrition_button = gr.Button("ניתוח דפוסי תזונה")
            nutrition_output = gr.Markdown(label="ערכים תזונתיים")
            
        with gr.TabItem("📊 ניתוח מקיף"):
            comprehensive_button = gr.Button("ניתוח מקיף")
            comprehensive_output = gr.Markdown(label="תובנות מקיפות")

    upload_button.click(fn=upload_json, inputs=[file_input], outputs=[processed_file, output_text])
    activity_button.click(fn=activity_analysis_summary, inputs=[mood_selector], outputs=[activity_output])
    medication_button.click(fn=medication_analysis_summary, inputs=[mood_selector], outputs=[medication_output])
    symptom_button.click(fn=symptom_analysis_summary, inputs=[mood_selector], outputs=[symptom_output])
    nutrition_button.click(fn=nutritional_analysis_summary, inputs=[], outputs=[nutrition_output])
    comprehensive_button.click(fn=comprehensive_analysis_summary, inputs=[mood_selector], outputs=[comprehensive_output])

# --- הפעלת האפליקציה ---

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.launch(server_name='0.0.0.0', server_port=port)
