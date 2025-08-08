import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load datasets
X_train = pd.read_csv("AI_Assignment/input/X_train.csv")
X_test = pd.read_csv("AI_Assignment/input/X_test.csv")
y_train = pd.read_csv("AI_Assignment/input/y_train.csv").values.ravel()
y_test = pd.read_csv("AI_Assignment/input/y_test.csv").values.ravel()

# Define class labels
target_names = ['edibl','poisonous']

# Initialize and train the ANN model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
                    activation='relu',            # Activation function
                    solver='adam',                # Optimizer
                    max_iter=500,                 # Number of training iterations
                    random_state=42)

mlp.fit(X_train, y_train)

# Predict on test set
y_pred = mlp.predict(X_test)

# Evaluate the performance
print("\n--- Artificial Neural Network (ANN) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
