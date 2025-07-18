import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('AI_Assignment/input/winequality-white.csv', sep=';', na_values=['null', 'Null', 'None', 'none', ' '])
# drop the quality attributes, because it is a target score
df = df.drop("quality", axis = 1)

# ------------------------------------------------ Remove duplicate ------------------------------------------------

# drop the duplicate row
df = df.drop_duplicates() 

# ------------------------------------------------ Handle Missing Value ------------------------------------------------

#drop the row with the missing value
df.dropna(inplace=True)

# ------------------------------------------------ Normalized data ------------------------------------------------

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler to df
normalized_data = scaler.fit_transform(df)

# Create a new Data Frame
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

