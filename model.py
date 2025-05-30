# 🚀 Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 📁 Load dataset
df = pd.read_csv(r"smote_balanced_data.csv")
df = df.drop(columns=["Favorite Color", "Birth Month"])
target = "What would you like to become when you grow up"

# 🧠 Merge classes
df[target] = df[target].replace({
    "Corporate Employee": "Conventional Career",
    "Government Officer": "Conventional Career"
})

# 🛠 Feature Engineering
df["risk_x_leadership"] = df["Risk-Taking Ability"] * df["Leadership Experience"]
df["cgpa_bucket"] = pd.cut(df["Academic Performance (CGPA/Percentage)"], [0, 60, 80, 100], labels=["Low", "Medium", "High"])
df["low_finance_high_risk"] = ((df["Risk-Taking Ability"] > 8) &
                               (df["Financial Stability - self/family (1 is low income and 10 is high income)"] < 5)).astype(int)
df["water_category"] = pd.cut(df["Daily Water Intake (in Litres)"], [0, 2, 4, 10], labels=["Low", "Medium", "High"])
df["academic_risk_profile"] = df["Academic Performance (CGPA/Percentage)"] * df["Risk-Taking Ability"]
df["leadership_to_siblings_ratio"] = df["Leadership Experience"] / (df["Number of Siblings"] + 1)
df["tech_finance_combo"] = df["Tech-Savviness"] * df["Financial Stability - self/family (1 is low income and 10 is high income)"]
df = pd.get_dummies(df, columns=["cgpa_bucket", "water_category"], drop_first=True)

# 👥 Behavioral Clustering (Personas)
cluster_features = [
    "Risk-Taking Ability", "Financial Stability - self/family (1 is low income and 10 is high income)",
    "Leadership Experience", "Academic Performance (CGPA/Percentage)", "Tech-Savviness"
]
kmeans = KMeans(n_clusters=4, random_state=42)
df["persona_cluster"] = kmeans.fit_predict(df[cluster_features])

# 🎯 Final features
features = [
    "Preferred Work Environment", "Risk-Taking Ability", "Age",
    "Financial Stability - self/family (1 is low income and 10 is high income)",
    "Preferred Subjects in Highschool/College", "Number of Siblings",
    "Participation in Extracurricular Activities", "Preferred Music Genre",
    "Leadership Experience", "Tech-Savviness", "Motivation for Career Choice",
    "risk_x_leadership", "low_finance_high_risk", "cgpa_bucket_Medium", "cgpa_bucket_High",
    "water_category_Medium", "water_category_High", "academic_risk_profile",
    "leadership_to_siblings_ratio", "tech_finance_combo", "persona_cluster"
]

X = df[features]
y = df[target]

# 🔠 Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🔀 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# ⚖️ SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 🌲 Tuned Random Forest
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
import joblib
joblib.dump(rf, 'model.joblib')


# 🔮 Predict
probs = rf.predict_proba(X_test)
initial_preds = np.argmax(probs, axis=1)

# ✅ Soft Correction
corrected_preds = initial_preds.copy()
for i, row in enumerate(probs):
    top_prob = np.max(row)
    second_class = np.argsort(row)[-2]
    second_prob = row[second_class]
    if top_prob < 0.37 and second_prob > 0.30:
        corrected_preds[i] = second_class

# 📊 Results
accuracy = accuracy_score(y_test, corrected_preds)
print(f"\n✅ Final Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, corrected_preds, target_names=label_encoder.classes_))

