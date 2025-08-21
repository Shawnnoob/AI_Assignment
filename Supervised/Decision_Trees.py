import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from Data_Preparation.Data_Preprocessing import X

# Filepath
path = "../"

X_train = pd.read_csv(f"{path}input/X_train.csv", delimiter=',')
X_test = pd.read_csv(f"{path}input/X_test.csv", delimiter=',')
y_train = pd.read_csv(f"{path}input/y_train.csv", delimiter=',')
y_test = pd.read_csv(f"{path}input/y_test.csv", delimiter=',')

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

# Visualize the Decision Tree
plt.figure(figsize=(14, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Edible', 'Poisonous'], rounded=True, fontsize=7)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig(f"{path}graphs/model_results/decision_tree.png")