import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from Data_Preparation.Data_Preprocessing import X

# Filepath
path = "../"

X_train = pd.read_csv(f"{path}input/X_train.csv", delimiter=',')
X_test = pd.read_csv(f"{path}input/X_test.csv", delimiter=',')
y_train = pd.read_csv(f"{path}input/y_train.csv", delimiter=',')
y_test = pd.read_csv(f"{path}input/y_test.csv", delimiter=',')

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

# Feature Importance (Optional, to check how important each feature is)
feature_importances = rf_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(f"{path}graphs/model_results/Random_Forest_Feature_Importance.png")