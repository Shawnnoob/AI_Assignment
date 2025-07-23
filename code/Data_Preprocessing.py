import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('../input/winequality-white.csv', sep=';', na_values=['null', 'Null', 'None', 'none', ' '])
df.rename(columns = {"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity",
                     "citric acid": "citric_acid", "residual sugar": "residual_sugar",
                     "chlorides": "chlorides", "free sulfur dioxide": "free_sulfur_dioxide",
                     "total sulfur dioxide": "total_sulfur_dioxide"}, inplace = True)
# drop the quality attributes, because it is a target score
df = df.drop("quality", axis = 1)

# ---------------------------- Remove duplicate and Missing Value -------------------------------

# drop the duplicate row
df = df.drop_duplicates()

# drop the row with the missing value
df.dropna(inplace=True)

# Save a copy of clean dataset for backup and data exploration purpose
df.to_csv("../input/clean_df.csv", index=False)

# ------------------------------------------ Normalized data --------------------------------

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler to df
normalized_data = scaler.fit_transform(df)

# Create a new normalized Data Frame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
#print(normalized_df)
normalized_df.to_csv("../input/normalized_df.csv", index=False) # Save normalized data