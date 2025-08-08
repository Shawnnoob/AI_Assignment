import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the training and testing sets
X_train = pd.read_csv("AI_Assignment/input/X_train.csv")
X_test = pd.read_csv("AI_Assignment/input/X_test.csv")
y_train = pd.read_csv("AI_Assignment/input/y_train.csv").values.ravel()  # Flatten to 1D
y_test = pd.read_csv("AI_Assignment/input/y_test.csv").values.ravel()

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=60)  # You can change k here (e.g. 3, 5, 7)
knn.fit(X_train, y_train)

# Predict the test set
y_pred = knn.predict(X_test)

# Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edibl','poisonous']))
