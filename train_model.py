import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('icwsmR_final_dataset_share.csv')

df['midnight_wt_total'] = df[['hour_0_wt', 'hour_1_wt', 'hour_2_wt', 'hour_3_wt', 'hour_4_wt', 'hour_5_wt']].sum(axis=1)
df['total_wt'] = df['morning_wt'] + df['noon_wt'] + df['afternoon_wt'] + df['evening_wt'] + df['midnight_wt_total']

df['midnight_ratio'] = df['midnight_wt_total'] / (df['total_wt'] + 1)

features = [
    'morning_wt', 'noon_wt', 'afternoon_wt', 'evening_wt', 
    'midnight_wt_total', 'total_wt', 'session_all_per_day', 'category_count_unique', 'age', 'midnight_ratio'
]
target = 'preds_3_label_criteria' 

X = df[features].fillna(0)
y = df[target].fillna('Unknown') 

print("Training pure Random Forest Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sample_weights = np.where((X_train['midnight_wt_total'] > 60), 5.0, 1.0)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train, sample_weight=sample_weights)

print("\n" + "="*50)
print("📊 MODEL EVALUATION METRICS")
print("="*50)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))
print("="*50 + "\n")

joblib.dump(clf, 'decision_support_model.joblib')
print("✅ Success: decision_support_model.joblib saved!")