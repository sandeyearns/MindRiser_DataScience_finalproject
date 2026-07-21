"""
Melbourne Housing Price Predictor - Streamlit App
--------------------------------------------------
Run locally with:
    streamlit run app.py


"""

import pickle
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Melbourne House Price Predictor", 
                   page_icon="🏠",
                    layout="centered")

# -----------------------------
# Load model artifact (cached so it only loads once)
# -----------------------------
@st.cache_resource
def load_artifact():
    with open("melbourne_house_price_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact

try:
    artifact = load_artifact()
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]
    needs_scaling = artifact["needs_scaling"]
    model_name = artifact["model_name"]
except FileNotFoundError:
    st.error(
        "Could not find 'melbourne_house_price_model.pkl'. "
        "Make sure it's in the same folder as app.py (run the notebook cell that saves it first)."
    )
    st.stop()

# -----------------------------
# Prediction function (same logic as in the notebook)
# -----------------------------
def predict_price(input_dict):
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    row_final = scaler.transform(row_encoded) if needs_scaling else row_encoded
    return model.predict(row_final)[0]

# -----------------------------
# Header
# -----------------------------
st.title("🏠 Melbourne House Price Predictor")
st.caption(f"Model in use: {model_name}")
st.write("Fill in the property details below and click **Predict Price**.")

# -----------------------------
# Input form
# -----------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.text_input("Suburb", value="Richmond")
        rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
        prop_type = st.selectbox("Type", options=["h", "u", "t"],
                                  help="h = house/cottage, u = unit/duplex, t = townhouse")
        method = st.selectbox("Method", options=["S", "SP", "PI", "VB", "SA"],
                               help="S=sold, SP=sold prior, PI=passed in, VB=vendor bid, SA=sold after auction")
        seller_g = st.text_input("SellerG (agent)", value="Nelson")
        distance = st.number_input("Distance from CBD (km)", min_value=0.0, value=5.2)
        postcode = st.number_input("Postcode", min_value=3000, max_value=3999, value=3121)
        bedroom2 = st.number_input("Bedroom2", min_value=0, max_value=10, value=3)
        bathroom = st.number_input("Bathroom", min_value=0, max_value=10, value=2)
        car = st.number_input("Car spots", min_value=0, max_value=10, value=1)

    with col2:
        landsize = st.number_input("Landsize (sqm)", min_value=0, value=450)
        building_area = st.number_input("Building Area (sqm)", min_value=0, value=150)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2005)
        council_area = st.text_input("Council Area", value="Yarra")
        lattitude = st.number_input("Latitude", value=-37.80, format="%.5f")
        longtitude = st.number_input("Longitude", value=144.99, format="%.5f")
        regionname = st.selectbox("Region Name", options=[
            "Northern Metropolitan", "Southern Metropolitan", "Western Metropolitan",
            "Eastern Metropolitan", "South-Eastern Metropolitan",
            "Northern Victoria", "Eastern Victoria", "Western Victoria"
        ])
        propertycount = st.number_input("Property Count (suburb)", min_value=0, value=4000)
        sale_year = st.number_input("Sale Year", min_value=2016, max_value=2026, value=2017)
        sale_month = st.number_input("Sale Month", min_value=1, max_value=12, value=6)

    submitted = st.form_submit_button("Predict Price")

# -----------------------------
# Prediction output
# -----------------------------
if submitted:
    new_property = {
        "Suburb": suburb,
        "Rooms": rooms,
        "Type": prop_type,
        "Method": method,
        "SellerG": seller_g,
        "Distance": distance,
        "Postcode": postcode,
        "Bedroom2": bedroom2,
        "Bathroom": bathroom,
        "Car": car,
        "Landsize": landsize,
        "BuildingArea": building_area,
        "YearBuilt": year_built,
        "CouncilArea": council_area,
        "Lattitude": lattitude,
        "Longtitude": longtitude,
        "Regionname": regionname,
        "Propertycount": propertycount,
        "SaleYear": sale_year,
        "SaleMonth": sale_month,
    }

    predicted_price = predict_price(new_property)
    st.success(f"### Predicted House Price: ${predicted_price:,.2f}")

    with st.expander("See input details"):
        st.json(new_property)

# & "C:\Program Files\Python313\python.exe" -m pip install scikit-learn xgboost streamlit
# second - & "C:\Program Files\Python313\python.exe" -m streamlit run app.py