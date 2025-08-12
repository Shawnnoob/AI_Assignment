import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Filepath
path = "../"

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
)

xgb_model.fit(X_train_sub, y_train_sub,
          eval_set=[(X_valid, y_valid)],
          verbose=True)

y_pred = xgb_model.predict(X_test)

# Show the important features used in model
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance":xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.head(10))

# Save model
joblib.dump(xgb_model, f'{path}Supervised/Gradient_Boosting.pkl')

# Evaluation
print("\n--- Gradient Boosting (XGBoost) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['edible','poisonous']))