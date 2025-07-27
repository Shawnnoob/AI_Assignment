import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X_train = pd.read_csv("../input/X_train.csv", delimiter=',')
X_test = pd.read_csv("../input/X_test.csv", delimiter=',')
y_train = pd.read_csv("../input/y_train.csv", delimiter=',')
y_test = pd.read_csv("../input/y_test.csv", delimiter=',')

# Split training data further into training + validation sets
X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model = XGBClassifier(
    objective='multi:softmax', # For class labels
    num_class=3, # 3 classes (low, normal, high)
    learning_rate=0.05,
    n_estimators=100, # Numbers of trees per round
    max_depth=7, # Max depth of trees
    subsample=0.8, # % of rows sampled per tree
    random_state=42,
    early_stopping_rounds=10,
)

model.fit(X_train_sub, y_train_sub,
          eval_set=[(X_valid, y_valid)],
          verbose=True)

predictions = model.predict(X_test)

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance":model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance_df.head(10))

# Evaluation
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions, target_names=['Low', 'Normal', 'High'])

print(conf_matrix)
print(class_report)