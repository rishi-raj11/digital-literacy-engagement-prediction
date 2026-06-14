import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("/Users/rishirajpurohit/Downloads/project/digital_literacy_dataset.csv")

# ==========================
# HANDLE MISSING VALUES
# ==========================

df['Education_Level'] = df['Education_Level'].fillna("Unknown")

# ==========================
# FEATURES & TARGET
# ==========================

features = [
    'Basic_Computer_Knowledge_Score',
    'Internet_Usage_Score',
    'Mobile_Literacy_Score',
    'Modules_Completed',
    'Average_Time_Per_Module',
    'Quiz_Performance',
    'Session_Count',
    'Adaptability_Score',
    'Feedback_Rating',
    'Skill_Application',
    'Overall_Literacy_Score'
]

target = "Engagement_Level"

# ==========================
# TRAIN MODEL
# ==========================

def train_model():

    X = df[features]

    le = LabelEncoder()
    y = le.fit_transform(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=========================")
    print("MODEL PERFORMANCE")
    print("=========================")

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
    y_test,
    y_pred,
    zero_division=0))


    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ==========================
    # FEATURE IMPORTANCE
    # ==========================

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n=========================")
    print("FEATURE IMPORTANCE")
    print("=========================")

    print(importance)

    # ==========================
    # SAVE MODEL
    # ==========================

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(le, open("label_encoder.pkl", "wb"))

    print("\nModel Saved Successfully!")

# ==========================
# USER INPUT
# ==========================

def get_user_input():

    print("\nEnter User Details:\n")

    basic = float(input("Basic Computer Knowledge Score: "))
    internet = float(input("Internet Usage Score: "))
    mobile = float(input("Mobile Literacy Score: "))
    modules = float(input("Modules Completed: "))
    avg_time = float(input("Average Time Per Module: "))
    quiz = float(input("Quiz Performance: "))
    sessions = float(input("Session Count: "))
    adaptability = float(input("Adaptability Score: "))
    feedback = float(input("Feedback Rating: "))
    skill = float(input("Skill Application: "))
    overall = float(input("Overall Literacy Score: "))

    return [[
        basic,
        internet,
        mobile,
        modules,
        avg_time,
        quiz,
        sessions,
        adaptability,
        feedback,
        skill,
        overall
    ]]

# ==========================
# PREDICTION SYSTEM
# ==========================

def run_prediction():

    model = pickle.load(open("model.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))

    while True:

        user_data = get_user_input()

        prediction = model.predict(user_data)[0]

        engagement = encoder.inverse_transform([prediction])[0]

        probability = model.predict_proba(user_data)[0]

        confidence = round(max(probability) * 100, 2)

        print("\n=========================")
        print("PREDICTION RESULT")
        print("=========================")

        print(f"\nPredicted Engagement Level : {engagement}")
        print(f"Confidence                : {confidence}%")

        if engagement == "High":
            print("Excellent learner engagement.")
        elif engagement == "Medium":
            print("Moderate learner engagement.")
        else:
            print("Low learner engagement. Intervention recommended.")

        again = input("\nCheck another user? (yes/no): ")

        if again.lower() != "yes":
            break

# ==========================
# MAIN MENU
# ==========================

choice = input(
    "\n1 -> Train Model\n"
    "2 -> Run Prediction\n\n"
    "Enter Choice: "
)

if choice == "1":
    train_model()

elif choice == "2":
    run_prediction()

else:
    print("Invalid Choice!")
    

