import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Data Visualization

pd.set_option('display.max_columns', None)

path = "AI_Assignment/"


# Load the dataset
df = pd.read_csv(f'{path}input/mushrooms.csv', sep = ',')



# --------------------------------------- Remove duplicate, Missing Value and delete the unrelated column ---------------------------------------
# drop the duplicate row
df = df.drop_duplicates()

# Define all missing value placeholders
missing_values = ["?", "null", "none", "None", "NONE"]

# Replace them with NaN
df.replace(missing_values, pd.NA, inplace=True)

# drop the row with the missing value
df.dropna(inplace = True)
print(df.describe())

df = df.drop(["veil-type", "cap-surface", "cap-shape", "gill-attachment", "gill-spacing", "stalk-shape", "stalk-root", "veil-color", "population"], axis = 1)

# Remove features
# cap-surface, cap-shape, gill-attachment, gill-spacing, stalk-shape, stalk-root, veil-type, veil-color, population,


# --------------------------------------- Mapping ---------------------------------------
# Mapping dictionary
mapping = {
    "class": {"p": 1, "e": 0},
    "cap-color": {"b": 0, "c": 1, "e": 2, "g": 3, "n": 4, "p": 5, "r": 6, "u": 7, "w": 8, "y": 9},
    "bruises": {"f": 0, "t": 1},
    "odor": {"a": 0, "c": 1, "f": 2, "l": 3, "m": 4, "n": 5, "p": 6, "s": 7, "y": 8},
    "gill-size": {"b": 0, "n": 1},
    "gill-color": {"b": 0, "e": 1, "g": 2, "h": 3, "k": 4, "n": 5, "o": 6, "p": 7, "r": 8, "u": 9, "w": 10, "y": 11},
    "stalk-surface-above-ring": {"f": 0, "k": 1, "s": 2, "y": 3},
    "stalk-surface-below-ring": {"f": 0, "k": 1, "s": 2, "y": 3},
    "stalk-color-above-ring": {"b": 0, "c": 1, "e": 2, "g": 3, "n": 4, "o": 5, "p": 6, "w": 7, "y": 8},
    "stalk-color-below-ring": {"b": 0, "c": 1, "e": 2, "g": 3, "n": 4, "o": 5, "p": 6, "w": 7, "y": 8},
    "ring-number": {"n": 0, "o": 1, "t": 2},
    "ring-type": {"e": 0, "f": 1, "l": 2, "n": 3, "p": 4},
    "spore-print-color": {"b": 0, "h": 1, "k": 2, "n": 3, "o": 4, "r": 5, "u": 6, "w": 7, "y": 8},
    "habitat": {"d": 0, "g": 1, "l": 2, "m": 3, "p": 4, "u": 5, "w": 6}
}

# Apply mapping to all columns
df_encoded = df.replace(mapping)

# Save a copy of clean dataset for backup and data exploration purpose

df_encoded.to_csv(f"{path}input/Complete_df.csv", index = False)
 

# --------------------------------------- Split into training and test set and balancing  ---------------------------------------
X = df_encoded.drop('class', axis = 1).copy()
y = df_encoded["class"].copy()
# split the dataset into training set = 80% and testing set = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.to_csv(f"{path}input/X_train.csv", index=False)
X_test.to_csv(f"{path}input/X_test.csv", index=False)
y_train.to_csv(f"{path}input/y_train.csv", index=False)
y_test.to_csv(f"{path}input/y_test.csv", index=False)