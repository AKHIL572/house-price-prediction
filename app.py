import matplotlib.pyplot as plt
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction App")
st.write("Enter property details to estimate the house price.")

# ---------------------------------------
# Load Model Artifacts
# ---------------------------------------

MODEL_PATH = "models/house_price_pipeline.pkl"
COLUMNS_PATH = "models/columns.pkl"

model_pipeline = joblib.load(MODEL_PATH)

with open(COLUMNS_PATH, "rb") as f:
    model_columns = pickle.load(f)

# Extract model from pipeline
model = model_pipeline.named_steps["model"]

# ---------------------------------------
# INPUT FORM
# ---------------------------------------

with st.form("prediction_form"):

    st.header("🏠 Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.25
        )
        floors = st.number_input("Floors", 1, 5, 1)

    with col2:
        sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 2000)
        sqft_lot = st.number_input("Lot Area (sqft)", 500, 100000, 5000)

    with col3:
        sqft_above = st.number_input("Sqft Above Ground", 300, 10000, 1500)
        sqft_basement = st.number_input("Basement Area", 0, 5000, 500)

    st.header("📍 Location")

    col4, col5, col6 = st.columns(3)

    with col4:
        lat = st.number_input("Latitude", value=47.5)
        long = st.number_input("Longitude", value=-122.2)

    with col5:
        zipcode = st.selectbox(
            "Zipcode",
            [
                98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008,
                98010, 98011, 98014, 98019, 98022, 98023, 98024, 98027,
                98028, 98029, 98030, 98031, 98032, 98033, 98034, 98038,
                98039, 98040, 98042, 98045, 98052, 98053, 98055, 98056,
                98058, 98059, 98065, 98070, 98072, 98074, 98075, 98077,
                98092, 98102, 98103, 98105, 98106, 98107, 98108, 98109,
                98112, 98115, 98116, 98117, 98118, 98119, 98122, 98125,
                98126, 98133, 98136, 98144, 98146, 98148, 98155, 98166,
                98168, 98177, 98178, 98188, 98198, 98199
            ]
        )

    with col6:
        waterfront = st.selectbox("Waterfront", [0, 1])
        view = st.slider("View Rating", 0, 4, 0)

    st.header("🏡 House Quality")

    col7, col8 = st.columns(2)

    with col7:
        condition = st.slider("Condition", 1, 5, 3)

    with col8:
        grade = st.slider("Grade", 1, 13, 7)

    st.header("🏘️ Neighborhood")

    col9, col10 = st.columns(2)

    with col9:
        sqft_living15 = st.number_input("Nearby Living Area", 300, 10000, 1800)

    with col10:
        sqft_lot15 = st.number_input("Nearby Lot Area", 500, 100000, 4500)

    st.header("🏗 House Age")

    house_age = st.number_input("House Age (years)", 0, 200, 20)

    submit = st.form_submit_button("Predict Price")

# ---------------------------------------
# Prediction
# ---------------------------------------

if submit:

    # Base input data
    input_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "house_age": house_age
    }

    input_df = pd.DataFrame([input_data])

    # ---------------------------------------
    # Zipcode Encoding
    # ---------------------------------------

    zipcode_col = f"zipcode_{zipcode}"

    for col in model_columns:
        if col.startswith("zipcode_"):
            input_df[col] = 0

    if zipcode_col in model_columns:
        input_df[zipcode_col] = 1

    # Align columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # ---------------------------------------
    # Prediction
    # ---------------------------------------

    log_price = model_pipeline.predict(input_df)[0]
    predicted_price = np.exp(log_price)

    # Price range (±10%)
    lower_price = predicted_price * 0.9
    upper_price = predicted_price * 1.1

    st.markdown("---")

    st.header("📊 Prediction Result")

    st.markdown(
        f"""
        ## 💰 Estimated House Price  
        # ${predicted_price:,.0f}
        """
    )

    st.write(
        f"Typical Range: **${lower_price:,.0f} – ${upper_price:,.0f}**"
    )

    st.markdown("---")

    # ---------------------------------------
    # Map Visualization
    # ---------------------------------------

    st.subheader("📍 House Location")

    map_df = pd.DataFrame({
        "lat": [lat],
        "lon": [long]
    })

    st.map(map_df)

    # ---------------------------------------
    # Feature Importance
    # ---------------------------------------

    if hasattr(model, "feature_importances_"):

        st.subheader("📈 Top Factors Affecting Price")

        importances = model.feature_importances_

        feat_imp = pd.DataFrame({
            "Feature": model_columns,
            "Importance": importances
        })

        feat_imp = feat_imp.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(feat_imp["Feature"], feat_imp["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=7.5)
        ax.set_title("Top Features Influencing Price", fontsize=9)

        # Tick label size
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        st.pyplot(fig, use_container_width=False)

    st.success("Prediction generated using the trained machine learning model.")
