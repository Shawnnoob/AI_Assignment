import pandas as pd
import time
import psutil, os
from sklearn.model_selection import train_test_split                
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Filepath
path = "AI_Assignment/"

X_train = pd.read_csv(f"{path}input/X_train.csv", delimiter=',')
X_test = pd.read_csv(f"{path}input/X_test.csv", delimiter=',')
y_train = pd.read_csv(f"{path}input/y_train.csv", delimiter=',')
y_test = pd.read_csv(f"{path}input/y_test.csv", delimiter=',')

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

process = psutil.Process(os.getpid())

before = process.memory_info().rss  # in bytes

start = time.time()

y_pred = rf_model.predict(X_test)

end = time.time()

after = process.memory_info().rss

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

# Feature Importance (Optional, to check how important each feature is)
feature_importances = rf_model.feature_importances_
features = X_train.columns

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{path}graphs/model_results/Random_Forest_Feature_Importance.png")

print(f"Prediction time: {end - start:.4f} seconds")
print(f"Memory Used in Prediction: {(after - before) / 1024**2:.4f} MB")



# --- Random Forest ---
# Accuracy: 1.0
# Confusion Matrix:
#  [[842   0]
#  [  0 783]]
# Classification Report:
#                precision    recall  f1-score   support

#       edible       1.00      1.00      1.00       842
#    poisonous       1.00      1.00      1.00       783

#     accuracy                           1.00      1625
#    macro avg       1.00      1.00      1.00      1625
# weighted avg       1.00      1.00      1.00      1625

# Prediction time: 0.0845 seconds
# Memory Used in Prediction: 0.1172 MB