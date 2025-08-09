import streamlit as st

# Attribute encoding dictionaries
encodings = {
    "cap-color": {"b": "buff", "c": "cinnamon", "e": "red", "g": "gray", "n": "brown", "p": "pink", "r": "green", "u": "purple", "w": "white", "y": "yellow"},
    "bruises": {"f": "no bruises", "t": "bruises"},
    "odor": {"a": "almond", "c": "creosote", "f": "foul", "l": "anise", "m": "musty", "n": "none", "p": "pungent", "s": "spicy", "y": "fishy"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"b": "buff", "e": "red", "g": "gray", "h": "chocolate", "k": "black", "n": "brown", "o": "orange", "p": "pink", "r": "green", "u": "purple", "w": "white", "y": "yellow"},
    "stalk-surface-above-ring": {"f": "fibrous", "k": "silky", "s": "smooth", "y": "scaly"},
    "stalk-surface-below-ring": {"f": "fibrous", "k": "silky", "s": "smooth", "y": "scaly"},
    "stalk-color-above-ring": {"b": "buff", "c": "cinnamon", "e": "red", "g": "gray", "n": "brown", "o": "orange", "p": "pink", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"b": "buff", "c": "cinnamon", "e": "red", "g": "gray", "n": "brown", "o": "orange", "p": "pink", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant"},
    "spore-print-color": {"b": "buff", "h": "chocolate", "k": "black", "n": "brown", "o": "orange", "r": "green", "u": "purple", "w": "white", "y": "yellow"},
    "habitat": {"d": "wood", "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste"}
}

st.title("Mushroom Attribute Selector")

# Model selection
model_choice = st.radio(
    "Choose a model for prediction:",
    ("K-Nearest Neighbors (KNN)", "Gradient Boosting", "Artificial Neural Network (ANN)")
)

user_selections = {}

# For each attribute, create a selectbox with code and meaning
for attr, code_map in encodings.items():
    options = [f"{code} = {meaning}" for code, meaning in code_map.items()]
    choice = st.selectbox(
        f"{attr.replace('-', ' ').title()}",
        options,
        key=attr
    )
    selected_code = choice.split(" = ")[0]
    user_selections[attr] = encodings[attr][selected_code]

st.write("---")
st.header("Encoded input for model:")
st.json(user_selections)
st.write(f"**Selected Model:** {model_choice}")

# Example: button to predict (insert model loading and prediction here)
if st.button("Predict"):
    st.info("Call your prediction model here with the above input!")