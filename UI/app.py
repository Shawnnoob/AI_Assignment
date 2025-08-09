import streamlit as st
import pandas as pd
import joblib

# Filepath
path = "AI_Assignment/"

# For encoding purpose
encodings = {
    "cap-color": {
        "Buff": 0, "Cinnamon": 1, "Red": 2, "Gray": 3, "Brown": 4,
        "Pink": 5, "Green": 6, "Purple": 7, "White": 8, "Yellow": 9
    },
    "bruises": {
        "No bruises": 0, "Bruises": 1
    },
    "odor": {
        "Almond": 0, "Creosote": 1, "Foul": 2, "Anise": 3,
        "Musty": 4, "None": 5, "Pungent": 6, "Spicy": 7, "Fishy": 8
    },
    "gill-size": {
        "Broad": 0, "Narrow": 1
    },
    "gill-color": {
        "Buff": 0, "Red": 1, "Gray": 2, "Chocolate": 3, "Black": 4, "Brown": 5,
        "Orange": 6, "Pink": 7, "Green": 8, "Purple": 9, "White": 10, "Yellow": 11
    },
    "stalk-surface-above-ring": {
        "Fibrous": 0, "Silky": 1, "Smooth": 2, "Scaly": 3
    },
    "stalk-surface-below-ring": {
        "Fibrous": 0, "Silky": 1, "Smooth": 2, "Scaly": 3
    },
    "stalk-color-above-ring": {
        "Buff": 0, "Cinnamon": 1, "Red": 2, "Gray": 3, "Brown": 4,
        "Orange": 5, "Pink": 6, "White": 7, "Yellow": 8
    },
    "stalk-color-below-ring": {
        "Buff": 0, "Cinnamon": 1, "Red": 2, "Gray": 3, "Brown": 4,
        "Orange": 5, "Pink": 6, "White": 7, "Yellow": 8
    },
    "ring-number": {
        "None": 0, "One": 1, "Two": 2
    },
    "ring-type": {
        "Evanescent": 0, "Flaring": 1, "Large": 2, "None": 3, "Pendant": 4
    },
    "spore-print-color": {
        "Buff": 0, "Chocolate": 1, "Black": 2, "Brown": 3, "Orange": 4,
        "Green": 5, "Purple": 6, "White": 7, "Yellow": 8
    },
    "habitat": {
        "Wood": 0, "Grasses": 1, "Leaves": 2, "Meadows": 3,
        "Paths": 4, "Urban": 5, "Waste": 6
    }
}

st.markdown(
    "<h1 style='text-align: center; font-family: cursive;'>🍄 Mushroom Classification App</h1>",
    unsafe_allow_html=True
)

# Load models 
knn_model = joblib.load(f'{path}/Supervised/K-NN.pkl')
gb_model = joblib.load(f'{path}/Supervised/Gradient_Boosting.pkl')
ann_model = joblib.load(f'{path}/Supervised/ANN.pkl')

# Model selection
model_choice = st.radio(
    "Choose a model for prediction:",
    ("K-Nearest Neighbors (KNN)", "Gradient Boosting", "Artificial Neural Network (ANN)")
)

if model_choice == "K-Nearest Neighbors (KNN)":
    model = knn_model
elif model_choice == "Gradient Boosting":
    model = gb_model
elif model_choice == "Artificial Neural Network (ANN)":
    model = ann_model
else:
    model = ["Unknown model"]

# for user input 
user_selections = {}

# For each attribute, create a selectbox with code and meaning
for attr, code_map in encodings.items():
    options = list(code_map.keys())     #create a list for user select
    choice = st.selectbox(
        f"{attr.replace('-', ' ').title()}",
        options,
        key=attr
    )
    selected_code = choice.split(" = ")[0]
    user_selections[attr] = encodings[attr][selected_code]

# show the data user select
st.write("---")
st.markdown(
    "<h1 style='text-align: center; font-family: cursive;'>📦 Encoded input for model :</h1>",
    unsafe_allow_html=True
)
# st.header("📦 Encoded input for model:")
st.json(user_selections)
st.write(f"**Selected Model:** {model_choice}")

# Convert dictionary to a single-row DataFrame
input_df = pd.DataFrame([user_selections])

# Example: button to predict (insert model loading and prediction here)
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]  # Assuming model.predict returns array-like
        if prediction == 0 :
            st.success("This mushroom is edible !!! ")
        else :   
            st.warning("The mushroom is poisonous !!!")
    except Exception as e:
        st.error(f"Error during prediction: {e}")