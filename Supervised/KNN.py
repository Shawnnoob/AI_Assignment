import pandas as pd
import joblib
import time
import psutil, os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Filepath
path = "AI_Assignment/"

# Load the training and testing sets
X_train = pd.read_csv(f"{path}input/X_train.csv")
X_test = pd.read_csv(f"{path}input/X_test.csv")
y_train = pd.read_csv(f"{path}input/y_train.csv").values.ravel()  # Flatten to 1D
y_test = pd.read_csv(f"{path}input/y_test.csv").values.ravel()

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=6)  # You can change k here (e.g. 3, 5, 7)
knn.fit(X_train, y_train)

process = psutil.Process(os.getpid())

before = process.memory_info().rss  # in bytes

#start time
start = time.time()

# Predict the test set
y_pred = knn.predict(X_test)

end = time.time()

after = process.memory_info().rss

joblib.dump(knn, f'{path}Supervised/K-NN.pkl')

# Evaluate the results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))

print(f"Prediction time: {end - start:.4f} seconds")
print(f"Memory Used in Prediction: {(after - before) / 1024**2:.4f} MB")



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

# Prediction time: 0.0210 seconds
# Memory Used in Prediction: 0.1523 MB
