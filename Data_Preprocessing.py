import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('AI_Assignment/input/winequality-white.csv', sep=';', na_values=['null', 'Null', 'None', 'none', ' '])
# Rename column names
df.rename(columns = {"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity",
                     "citric acid": "citric_acid", "residual sugar": "residual_sugar",
                     "chlorides": "chlorides", "free sulfur dioxide": "free_sulfur_dioxide",
                     "total sulfur dioxide": "total_sulfur_dioxide"}, inplace = True)

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
preprocessing_data = scaler.fit_transform(df)

# Create a new Data Frame
preprocessing_data = pd.DataFrame(preprocessing_data, columns=df.columns)

# drop the quality attributes, because it is a target score
preprocessing_data = preprocessing_data.drop("quality", axis = 1)

print(preprocessing_data)
print(df)