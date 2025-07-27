import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('AI_Assignment/input/winequality-white.csv', sep = ';', na_values = ['null', 'Null', 'None', 'none', ' '])

# rename all the attributes
df.rename(columns = {"fixed acidity": "Fixed_Acidity", "volatile acidity": "Volatile_Acidity",
                     "citric acid": "Citric_Acid", "residual sugar": "Residual_Sugar",
                     "chlorides": "Chlorides", "free sulfur dioxide": "Free_Sulfur_Dioxide",
                     "total sulfur dioxide": "Total_Sulfur_Dioxide", "alcohol": "Alcohol", 
                     "sulphates" : "Sulphates"}, inplace = True)




# --------------------------------------- Generate New Attributes for Quality ---------------------------------------

def map_quality_label(q):
    if q <= 4:
        return 'Low' # if quality is small than 4 is low quality
    elif q <= 7:
        return 'Normal'# if quality is small or equal than 6 is normal quality
    else:
        return 'High'# if quality is large than 6 is high quality

df['Quality'] = df['quality'].apply(map_quality_label)




# --------------------------------------- Remove duplicate and Missing Value ---------------------------------------

# drop the duplicate row
df = df.drop_duplicates()

# drop the row with the missing value
df.dropna(inplace = True)

# Save a copy of clean dataset for backup and data exploration purpose
df.to_csv("AI_Assignment/input/clean_df.csv", index = False)




# --------------------------------------- Mixed the Attribute -----------------------------------------

# since the Residual_Sugar and density have the high correlation, so i mixed it
df["Sugar_Density"] = df["Residual_Sugar"] * df["density"]

df = df.drop(['quality', 'Residual_Sugar', 'density'], axis = 1)

df.to_csv("AI_Assignment/input/complete_df.csv", index = False)




# --------------------------------------- Normalized data ---------------------------------------

# drop the quality first because string cannot normalized
Nor_df = df.drop("Quality", axis = 1)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler to df
normalized_data = scaler.fit_transform(Nor_df)

# Create a new normalized Data Frame
normalized_df = pd.DataFrame(normalized_data, columns = Nor_df.columns)

#print(normalized_df)
normalized_df.to_csv("AI_Assignment/input/normalized_df.csv", index=False)




# --------------------------------------- Split into training and test set and balancing  ---------------------------------------

# get the type of value of the quality then encode in numerical
labels = df['Quality'].values

le = LabelEncoder()
y = le.fit_transform(labels)  # to become 0, 1, 2
X = normalized_df

# split the dataset into training set = 80% and testing set = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#oversampling with balancing the data
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

