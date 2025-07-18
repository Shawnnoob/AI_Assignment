import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns # Data Visualization

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

ori_df =pd.read_csv("AI_Assignment/input/winequality-white.csv", delimiter=';')
df = ori_df.copy()
# Rename column names
df.rename(columns = {"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity",
                     "citric acid": "citric_acid", "residual sugar": "residual_sugar",
                     "chlorides": "chlorides", "free sulfur dioxide": "free_sulfur_dioxide",
                     "total sulfur dioxide": "total_sulfur_dioxide"}, inplace = True)

#df.info()
#print(df.describe())

fig, axes = plt.subplots(1, 2, figsize = (30, 10)) # 1st plot
sns.histplot(ax = axes[0], x = df["fixed_acidity"],
             bins = 10,
             kde = True,
             cbar = True,
             color = "#CA96EC").set(title = "Distribution of 'fixed_acidity'")

sns.histplot(ax = axes[1], x = df["volatile_acidity"],
             bins = 10,
             cbar = True,
             kde = True,
             color = "#A163CF").set(title = "Distribution of 'volatile_acidity'")
plt.savefig("dist_FixedAcidity_VolatileAcidity.png", dpi=300) # Saves 1st plot as png

fig, axes = plt.subplots(1, 2, figsize=(30, 10)) # 2nd plot
sns.histplot(ax = axes[0], x = df["citric_acid"],
             bins = 10,
             kde = True,
             cbar = True,
             color = "#29066B").set(title = "Distribution of 'citric_acid'")

sns.histplot(ax = axes[1], x = df["alcohol"],
             bins = 10,
             kde = True,
             cbar = True,
             color = "#641811").set(title = "Distribution of 'alcohol'")
plt.savefig("dist_CitricAcid_Alcohol.png", dpi=300) # Saves 2nd plot as png

fig, axes = plt.subplots(1, 2, figsize=(30, 10)) # 3rd plot
sns.histplot(ax = axes[0], x = df["residual_sugar"],
             bins = 10,
             kde = True,
             cbar = True,
             color = "#EB548C").set(title = "Distribution of 'residual_sugar'")

sns.histplot(ax = axes[1], x = df["chlorides"],
             bins = 10,
             kde = True,
             cbar = True,
             color = "#EC96E0").set(title = "Distribution of 'chlorides'")
plt.savefig("dist_ResidualSugar_Chlorides.png", dpi=300) # Saves 3rd plot as png