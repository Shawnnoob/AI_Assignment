import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)

# Loading the dataset
data = pd.read_csv("../AI_Assignment/marketing_campaign.csv", sep="\t")
print("Number of records:", len(data))
# print(data.head())

# Information on features
#data.info()

# Remove null values
data = data.dropna()
print("Remaining records with null value removed:", len(data))

# Parse 'Dt_Customer' to DateTime, create feature 'Customer_Days' to indicate the number of days
# a customer is registered, relative to the most recent customer in record.
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format = "%d-%m-%Y")
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)
# Dates of the newest and oldest recorded customer
#print("The newest customer's enrolment date:",max(dates))
#print("The oldest customer's enrolment date:",min(dates))
d1 = max(dates)
days = [(d1 - d).days for d in dates] # newest date - current date for all records
data["Customer_Days"] = days
data["Customer_Days"] = pd.to_numeric(data["Customer_Days"], errors="coerce")

# Show unique values in categorical features
#print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
#print("Total categories in the feature Education:\n", data["Education"].value_counts())

# Feature Engineering
# Age of customer (based on 2021)
data["Age"] = 2021 - data["Year_Birth"]

# Total spending on various items
data["Total_Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + \
                      data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

# Marital status: Partner/Alone
data["Marital_Status"] = data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner",
                                                         "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone",
                                                         "Divorced":"Alone", "Single":"Alone",})

# Total children in household
data["Children"] = data["Kidhome"] + data["Teenhome"]

# Total members in household
data["Family_Size"] = data["Marital_Status"].replace({"Alone": 1, "Partner": 2}) + data["Children"]

# Parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0) #Condition,value_if_true,value_if_false

# Education level
data["Education"] = data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate",
                                               "Graduation":"Graduate", "Master":"Postgraduate",
                                               "PhD":"Postgraduate"})

# Enhance clarity
data = data.rename(columns = {"MntWines": "Wines", "MntFruits":"Fruits",
                              "MntMeatProducts":"Meat", "MntFishProducts":"Fish",
                              "MntSweetProducts":"Sweets", "MntGoldProds":"Gold"})

# Dropping redundant features
drop_features = ["Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(drop_features, axis=1)
print("Dataset after feature engineering:")
print(data.head())
#print(data.describe())