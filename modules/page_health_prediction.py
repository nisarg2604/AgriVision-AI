"""
Page: page_health_prediction.py
Purpose: Logistic Regression-based crop health prediction (Healthy / Unhealthy).
"""

import streamlit as st
import pandas as pd
from modules.models import predict_health


# Dropdown options (must match dataset generation constants)
CROP_TYPES    = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Potato", "Tomato"]
SOIL_TYPES    = ["Loamy", "Sandy", "Clay", "Silt", "Peaty", "Chalky", "Saline"]
LEAF_COLORS   = ["Green", "Yellow", "Brown", "Pale Green", "Dark Green", "Reddish"]
GROWTH_STAGES = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
REGIONS       = ["North", "South", "East", "West", "Central"]
SEASONS       = ["Kharif", "Rabi", "Zaid", "Summer", "Winter"]
PEST_OPT      = ["Yes", "No"]
FERTILIZERS   = ["Urea", "DAP", "NPK", "Organic", "Compost", "None"]
WATER_Q       = ["Good", "Moderate", "Poor"]
AIR_Q         = ["Clean", "Moderate", "Polluted"]
STRESS_L      = ["None", "Mild", "Moderate", "High"]
SEVERITY_L    = ["None", "Low", "Moderate", "High", "Severe"]


def show(df: pd.DataFrame, models: dict) -> None:
    st.title("🩺 Crop Health Prediction")
    st.markdown(
        "Enter crop and environmental parameters below. "
        "The **Logistic Regression** model will predict whether the crop is **Healthy** or **Unhealthy**."
    )
    st.markdown("---")

    with st.form("health_form"):
        st.subheader("🌿 Crop Vital Signs & Physical Features")

        col1, col2 = st.columns(2)

        with col1:
            crop_type      = st.selectbox("Crop Type",        CROP_TYPES)
            leaf_color     = st.selectbox("Leaf Color",       LEAF_COLORS)
            growth_stage   = st.selectbox("Growth Stage",     GROWTH_STAGES)
            stress_level   = st.selectbox("Current Stress Level", STRESS_L)
            soil_type      = st.selectbox("Soil Type",        SOIL_TYPES)

        with col2:
            ndvi_index     = st.slider("NDVI Index (Greenness)", 0.1, 0.9, 0.65, 0.01)
            chlorophyll    = st.slider("Chlorophyll Level (SPAD)", 10.0, 80.0, 45.0, 0.5)
            soil_moisture  = st.slider("Soil Moisture (%)",  10.0,  80.0, 45.0, 1.0)
            leaf_area      = st.slider("Leaf Area (cm²)",      5.0, 60.0, 30.0, 0.5)
            air_quality    = st.selectbox("Local Air Quality", AIR_Q)

        with st.expander("🛠️ Advanced Physical Parameters (Optional Defaults)"):
            c3, c4 = st.columns(2)
            with c3:
                leaf_spots     = st.slider("Leaf Spots (count)",  0,    20,   2)
                pest_presence  = st.selectbox("Pest Presence",    PEST_OPT, index=1)
            with c4:
                water_quality  = st.selectbox("Water Quality",  WATER_Q, index=0)
                severity       = st.selectbox("Disease Severity", SEVERITY_L, index=0)

        submitted = st.form_submit_button("🩺 Predict Crop Health", use_container_width=True)

    if submitted:
        # Use intelligent defaults for environmental/production features
        input_data = pd.DataFrame([{
            "Crop_Type"        : crop_type,
            "Soil_Type"        : soil_type,
            "Temperature"      : 26.5,  # Default
            "Humidity"         : 65.0,  # Default
            "Rainfall"         : 850.0, # Default
            "pH_Level"         : 6.5,   # Default
            "Nitrogen"         : 80.0,  # Default
            "Phosphorus"       : 45.0,  # Default
            "Potassium"        : 100.0, # Default
            "Leaf_Color"       : leaf_color,
            "Leaf_Spots"       : leaf_spots,
            "Disease_Severity" : severity,
            "Irrigation_Level" : 60.0,  # Default
            "Sunlight_Hours"   : 8.5,   # Default
            "Growth_Stage"     : growth_stage,
            "Pest_Presence"    : pest_presence,
            "Fertilizer_Used"  : "NPK", # Default
            "Region"           : "North", # Default
            "Season"           : "Kharif", # Default
            "Water_Quality"    : water_quality,
            "Air_Quality"      : air_quality,
            "Soil_Moisture"    : soil_moisture,
            "NDVI_Index"       : ndvi_index,
            "Stress_Level"     : stress_level,
            "Leaf_Area"        : leaf_area,
            "Chlorophyll_Level": chlorophyll,
        }])

        with st.spinner("Running Logistic Regression model…"):
            label, confidence = predict_health(models, input_data)

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if "Healthy" in label:
                st.success(f"## {label}")
                st.metric("Model Confidence", f"{confidence*100:.1f}%")
            else:
                st.error(f"## {label}")
                st.metric("Model Confidence", f"{confidence*100:.1f}%")

        with col_r2:
            st.progress(int(confidence * 100))
            if "Unhealthy" in label:
                st.warning(
                    "⚠️ **Recommended Actions:**\n\n"
                    "- Inspect leaves for visible disease symptoms\n"
                    "- Check soil nutrient levels and pH balance\n"
                    "- Review irrigation frequency and water quality\n"
                    "- Consider targeted fungicide or pesticide application\n"
                    "- Consult the **Recommendation System** tab for tailored advice"
                )
            else:
                st.info(
                    "✅ **Maintenance Tips:**\n\n"
                    "- Continue current crop management practices\n"
                    "- Perform regular field scouting\n"
                    "- Monitor NDVI and chlorophyll levels periodically"
                )

        # Download report
        report_df = input_data.copy()
        report_df["Prediction"] = label
        report_df["Confidence"] = f"{confidence*100:.1f}%"
        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=report_df.to_csv(index=False).encode(),
            file_name="health_prediction_report.csv",
            mime="text/csv",
        )
