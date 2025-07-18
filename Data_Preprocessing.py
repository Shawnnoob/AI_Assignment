import pandas as pd

# Load the dataset
df = pd.read_csv('AI_Assignment/input/winequality-white.csv')


# ------------------------------------------------ Remove duplicate ------------------------------------------------

# drop the duplicate row
df = df.drop_duplicates() 

# ------------------------------------------------ Handle Missing Value ------------------------------------------------

print(df.isnull().sum())