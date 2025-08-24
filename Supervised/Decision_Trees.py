import pandas as pd
import joblib
import time
import psutil, os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Filepath
path = "AI_Assignment/"

X_train = pd.read_csv(f"{path}input/X_train.csv", delimiter=',')
X_test = pd.read_csv(f"{path}input/X_test.csv", delimiter=',')
y_train = pd.read_csv(f"{path}input/y_train.csv", delimiter=',')
y_test = pd.read_csv(f"{path}input/y_test.csv", delimiter=',')

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

process = psutil.Process(os.getpid())

before = process.memory_info().rss  # in bytes

#start time
start = time.time()

y_pred = dt_model.predict(X_test)

end = time.time()

after = process.memory_info().rss

# Save model
joblib.dump(dt_model, f'{path}Supervised/Decision_Tree.pkl')

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

# Visualize the Decision Tree
plt.figure(figsize=(14, 10))
plot_tree(dt_model, filled=True, feature_names=X_train.columns, class_names=['Edible', 'Poisonous'], rounded=True, fontsize=7)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig(f"{path}graphs/model_results/decision_tree.png")

print(f"Prediction time: {end - start:.4f} seconds")
print(f"Memory Used in Prediction: {(after - before) / 1024**2:.4f} MB")


# --- Decision Tree ---
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

# Prediction time: 0.0012 seconds
# Memory Used in Prediction: 0.0234 MB