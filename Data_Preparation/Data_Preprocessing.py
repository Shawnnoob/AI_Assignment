import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Data Visualization

pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('../input/winequality-white.csv', sep = ';')

# rename all the attributes
df.rename(columns = {"fixed acidity": "Fixed_Acidity", "volatile acidity": "Volatile_Acidity",
                     "citric acid": "Citric_Acid", "residual sugar": "Residual_Sugar",
                     "chlorides": "Chlorides", "free sulfur dioxide": "Free_Sulfur_Dioxide",
                     "total sulfur dioxide": "Total_Sulfur_Dioxide", "alcohol": "Alcohol", 
                     "sulphates": "Sulphates", "density": "Density"}, inplace = True)


# --------------------------------------- Encode Quality ---------------------------------------
def map_quality_label(q):
    if q <= 4:
        return 'Low' # if quality is smaller than 4 is low quality
    elif q <= 7:
        return 'Normal'# if quality is small or equal than 7 is normal quality
    else:
        return 'High'# if quality is larger than 7 is high quality

df['Quality'] = df['quality'].apply(map_quality_label)

# Manual label mapping
quality_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
df['Quality'] = df['Quality'].map(quality_mapping)


# --------------------------------------- Remove duplicate and Missing Value ---------------------------------------
# drop the duplicate row
df = df.drop_duplicates()

# drop the row with the missing value
df.dropna(inplace = True)

# Save a copy of clean dataset for backup and data exploration purpose
df.to_csv("../input/clean_df.csv", index = False)


# --------------------------------------- Combine Features Using PCA --------------------------------
# Select features to combine
combine = df[["Density", "Residual_Sugar"]]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combine)

<<<<<<< HEAD
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
=======
# --------------------------------------- Mixed the Attribute -----------------------------------------
>>>>>>> baa2bd125609fa0a810e1fad3607cde97b65d9ff

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(pca.components_)

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.4)
plt.title('PCA of Density and Residual Sugar')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% Variance)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% Variance)')
plt.grid(True)
plt.tight_layout()
plt.savefig("../graphs/PCA_Density_Sugar.png")

# Manually apply PC1 formula (loadings are equal)
pc1 = X_scaled @ np.array([0.7071, 0.7071])  # Dot product for PC1

# Add to DataFrame
df['sugar_density_pca'] = pc1

# To check and compare result
df.to_csv("../input/mix_df.csv", index = False)


# --------------------------------------- Normalized data ---------------------------------------
# drop uneccessary features
Nor_df = df.drop(["Quality", "quality", "Density", "Residual_Sugar"], axis = 1)

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(Nor_df)

# Create a new normalized Data Frame
normalized_df = pd.DataFrame(normalized_data, columns = Nor_df.columns)

#print(normalized_df)
normalized_df.to_csv("../input/normalized_df.csv", index=False)


# --------------------------------------- Split into training and test set and balancing  ---------------------------------------
X = normalized_df.copy()
y = df["Quality"].copy()
# split the dataset into training set = 80% and testing set = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#oversampling with balancing the data
# sm = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

X_train.to_csv("../input/X_train.csv", index=False)
X_test.to_csv("../input/X_test.csv", index=False)
y_train.to_csv("../input/y_train.csv", index=False)
y_test.to_csv("../input/y_test.csv", index=False)