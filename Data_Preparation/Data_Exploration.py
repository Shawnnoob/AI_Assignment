import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns # Data Visualization
from scipy import stats

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

CSV_PATH = "../input/mushrooms.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)
print(f"Loaded {CSV_PATH} with shape {df.shape}\n")

# Overview
print(f"Overview of dataset:\n{df.describe()}\n")
df.info()


# FEATURE VS CLASS PLOTS
# Folder to save plots
output_dir = "../graphs/feature_vs_class_plots"
os.makedirs(output_dir, exist_ok=True)

# Identify the target column (class)
target_col = "class"  # Change if your label column is named differently
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset!")

# Loop through all features except the target
for col in df.columns:
    if col == target_col:
        continue

    plt.figure(figsize=(8, 5))

    # Create cross-tabulation between feature and class
    ct = pd.crosstab(df[col], df[target_col])

    # Normalize by row to get percentages per feature value
    ct_percent = ct.div(ct.sum(axis=1), axis=0) * 100

    # Plot as side-by-side bar chart
    ct_percent.plot(kind="bar", ax=plt.gca())
    plt.title(f"Distribution of '{col}' compared to '{target_col}'")
    plt.ylabel("Percentage (%)")
    plt.xlabel(col)
    plt.legend(title=target_col)
    plt.tight_layout()

    # Save plot
    filename = f"{col}_vs_{target_col}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"Plots saved in: {output_dir}")

# Convert categorical columns to category dtype
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype('category')


# CORRELATION MATRIX
# Convert categorical columns to category dtype
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype('category')

# Function to calculate Cramér's V for categorical variables
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = stats.chi2_contingency(confusion_matrix) #Chi-square test
    # Calculate Cramer's V statistic (chi-square
    return np.sqrt(chi2 / (confusion_matrix.sum().sum() * min(confusion_matrix.shape) - 1))

# Calculate Cramér's V correlation for categorical features
cat_cols = df.select_dtypes(include=["category"]).columns # select only category cols
corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols) # builds DataFrame

# Fill the correlation matrix with Cramér's V values, loop through all pairs of cat cols
for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 == col2: # When comparing same category, fill in 1
            corr_matrix.loc[col1, col2] = 1.0
        elif pd.isna(corr_matrix.loc[col1, col2]): # Calculate Cramer's V
            corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Matrix of Features (Cramér's V)")
plt.tight_layout()
plt.savefig("../graphs/Correlation_Matrix.png")