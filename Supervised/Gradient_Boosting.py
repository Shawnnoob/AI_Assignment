import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil, os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Filepath
path = "AI_Assignment/"

X_train = pd.read_csv(f"{path}input/X_train.csv", delimiter=',')
X_test = pd.read_csv(f"{path}input/X_test.csv", delimiter=',')
y_train = pd.read_csv(f"{path}input/y_train.csv", delimiter=',')
y_test = pd.read_csv(f"{path}input/y_test.csv", delimiter=',')

# Split training data further into training + validation sets
X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

xgb_model = XGBClassifier(
    objective='binary:logistic', # For binary classification
    n_estimators=80, # Numbers of trees per round
    max_depth=5, # Max depth of trees
    subsample=0.8, # % of rows sampled per tree
    random_state=42,
    early_stopping_rounds=10,
    eval_metric="logloss",
)

xgb_model.fit(X_train_sub, y_train_sub,
          eval_set=[(X_train_sub, y_train_sub), (X_valid, y_valid)],
          verbose=False)

process = psutil.Process(os.getpid())

before = process.memory_info().rss  # in bytes

#start time
start = time.time()

y_pred = xgb_model.predict(X_test)

end = time.time()

after = process.memory_info().rss

# Show the important features used in model
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance":xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.head(10))

# Evaluation
print("\n--- Gradient Boosting (XGBoost) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

print(f"Prediction time: {end - start:.4f} seconds")
print(f"Memory Used in Prediction: {(after - before) / 1024**2:.4f} MB")

# Get evaluation results
results = xgb_model.evals_result()

# Plot training vs validation logloss
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8,6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('XGBoost Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{path}graphs/overfitting/gradient_boosting_overfitting.png")
plt.show()

# --- Gradient Boosting (XGBoost) ---
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

# Prediction time: 0.0118 seconds
# Memory Used in Prediction: 0.0195 MB