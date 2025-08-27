import pandas as pd
import time
import psutil, os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Filepath
path = "AI_Assignment/"

# Load datasets
X_train = pd.read_csv(f"{path}input/X_train.csv")
X_test = pd.read_csv(f"{path}input/X_test.csv")
y_train = pd.read_csv(f"{path}input/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{path}input/y_test.csv").values.ravel()

# Define class labels
target_names = ['edible','poisonous']

# Training parameters
epochs = 50
train_scores = []
test_scores = []

# Initialize and train the ANN model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
                    activation='relu',            # Activation function
                    solver='adam',                # Optimizer
                    max_iter=1,                  # train 1 epoch at a time
                    warm_start=True,            # continue training instead of reinitializing
                    random_state=42)

mlp.fit(X_train, y_train)

process = psutil.Process(os.getpid())

before = process.memory_info().rss  # in bytes

#start time
start = time.time()

# Train over epochs and record accuracy
for _ in range(epochs):
    mlp.fit(X_train, y_train)
    train_scores.append(mlp.score(X_train, y_train))
    test_scores.append(mlp.score(X_test, y_test))

# Predict on test set
y_pred = mlp.predict(X_test)

end = time.time()

after = process.memory_info().rss

# Evaluate the performance
print("\n--- Artificial Neural Network (ANN) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

print(f"Prediction time: {end - start:.4f} seconds")
print(f"Memory Used in Prediction: {(after - before) / 1024**2:.4f} MB")

# --- Plot Training vs Test Accuracy ---
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_scores, label='Training Accuracy', color='blue')
plt.plot(range(1, epochs+1), test_scores, label='Test Accuracy', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(f"{path}graphs/overfitting/ann_overfitting.png")
plt.show()


# --- Artificial Neural Network (ANN) ---
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

# Prediction time: 0.0035 seconds
# Memory Used in Prediction: 2.7930 MB