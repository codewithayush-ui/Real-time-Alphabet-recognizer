import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

os.makedirs('models', exist_ok=True)
data_path = 'data/alphabet_data.csv'

# Load extracted landmark features
df = pd.read_csv(data_path)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Get unique labels and create a map
unique_classes = sorted(list(set(y)))
label_map = {int(i): str(label) for i, label in enumerate(unique_classes)}
with open('models/classes.json', 'w') as f:
    json.dump(label_map, f, indent=4)

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest Classifier on landmark features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

# Save model
joblib.dump(model, 'models/alphabet_model.pkl')
print("[DONE] Model saved to models/alphabet_model.pkl")