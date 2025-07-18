import pandas as pd

# Load the dataset
df = pd.read_csv('AI_Assignment/input/winequality-white.csv')


# ------------------------------------------------ Remove duplicate ------------------------------------------------

print("Duplicate rows:", df.duplicated().sum()) # Check how many duplicate rows exist
