"""
Page: page_disease_classification.py
Purpose: Random Forest multi-class disease classification.
"""

import streamlit as st
import pandas as pd
from modules.models import predict_disease


CROP_TYPES    = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Potato", "Tomato"]
SOIL_TYPES    = ["Loamy", "Sandy", "Clay", "Silt", "Peaty", "Chalky", "Saline"]
LEAF_COLORS   = ["Green", "Yellow", "Brown", "Pale Green", "Dark Green", "Reddish"]
GROWTH_STAGES = ["Germination", "Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]
REGIONS       = ["North", "South", "East", "West", "Central"]
SEASONS       = ["Kharif", "Rabi", "Zaid", "Summer", "Winter"]
FERTILIZERS   = ["Urea", "DAP", "NPK", "Organic", "Compost", "None"]
WATER_Q       = ["Good", "Moderate", "Poor"]
AIR_Q         = ["Clean", "Moderate", "Polluted"]
STRESS_L      = ["None", "Mild", "Moderate", "High"]
SEVERITY_L    = ["None", "Low", "Moderate", "High", "Severe"]

DISEASE_EMOJIS = {
    "Blight"      : "🟤",
    "Rust"        : "🟠",
    "Mildew"      : "⚪",
    "Mosaic Virus": "🟡",
    "Root Rot"    : "⚫",
    "Leaf Spot"   : "🟣",
    "Healthy"     : "🟢",
}

DISEASE_TIPS = {
    "Blight"      : "Apply copper-based fungicide. Ensure proper field drainage.",
    "Rust"        : "Use Propiconazole fungicide. Remove and burn infected leaves.",
    "Mildew"      : "Spray diluted neem oil. Improve canopy air circulation.",
    "Mosaic Virus": "No chemical cure. Remove infected plants. Control aphid vectors.",
    "Root Rot"    : "Treat with Trichoderma biofungicide. Avoid waterlogging.",
    "Leaf Spot"   : "Apply Chlorothalonil. Remove heavily infected leaves.",
    "Healthy"     : "No disease detected. Maintain regular crop management practices.",
}


def show(df: pd.DataFrame, models: dict) -> None:
    st.title("🦠 Disease Classification")
    st.markdown(
        "Uses a **Random Forest Classifier** trained on 10,000 records to predict "
        "the specific disease affecting your crop."
    )
    st.markdown("---")

    with st.form("disease_form"):
        st.subheader("📋 Disease Symptoms & Environmental Triggers")
        c1, c2 = st.columns(2)

        with c1:
            crop_type      = st.selectbox("Crop Type",        CROP_TYPES)
            leaf_color     = st.selectbox("Leaf Color",       LEAF_COLORS)
            leaf_spots     = st.slider("Visible Leaf Spots",  0,    20,    8)
            severity       = st.selectbox("Disease Severity", SEVERITY_L)
            growth_stage   = st.selectbox("Growth Stage",     GROWTH_STAGES)

        with c2:
            temperature    = st.slider("Temperature (°C)",    15.0, 45.0, 28.0)
            humidity       = st.slider("Humidity (%)",        30.0, 95.0, 70.0)
            rainfall       = st.slider("Rainfall (mm)",      200.0, 2000.0, 700.0)
            pest_presence  = st.selectbox("Pest Presence",    ["Yes","No"])
            stress_level   = st.selectbox("Stress Level",  STRESS_L)

        with st.expander("🛠️ Advanced Environmental Filters (Defaults)"):
            c3, c4 = st.columns(2)
            with c3:
                water_quality  = st.selectbox("Water Quality", WATER_Q, index=1)
                air_quality    = st.selectbox("Air Quality",   AIR_Q, index=1)
            with c4:
                soil_type      = st.selectbox("Soil Type",     SOIL_TYPES, index=0)
                region         = st.selectbox("Region",        REGIONS, index=0)

        submitted = st.form_submit_button("🔬 Classify Disease", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Crop_Type"        : crop_type,
            "Soil_Type"        : soil_type,
            "Temperature"      : temperature,
            "Humidity"         : humidity,
            "Rainfall"         : rainfall,
            "pH_Level"         : 6.2,     # Default
            "Nitrogen"         : 70.0,    # Default
            "Phosphorus"       : 35.0,    # Default
            "Potassium"        : 80.0,    # Default
            "Leaf_Color"       : leaf_color,
            "Leaf_Spots"       : leaf_spots,
            "Disease_Severity" : severity,
            "Irrigation_Level" : 55.0,    # Default
            "Sunlight_Hours"   : 7.0,     # Default
            "Growth_Stage"     : growth_stage,
            "Pest_Presence"    : pest_presence,
            "Fertilizer_Used"  : "NPK",   # Default
            "Region"           : region,
            "Season"           : "Rabi",  # Default
            "Water_Quality"    : water_quality,
            "Air_Quality"      : air_quality,
            "Soil_Moisture"    : 40.0,    # Default
            "NDVI_Index"       : 0.45,    # Default
            "Stress_Level"     : stress_level,
            "Leaf_Area"        : 25.0,    # Default
            "Chlorophyll_Level": 38.0,    # Default
        }])

        with st.spinner("Running Random Forest classifier…"):
            disease, confidence = predict_disease(models, input_df)

        st.markdown("---")
        st.subheader("🎯 Classification Result")

        emoji = DISEASE_EMOJIS.get(disease, "❓")
        tip   = DISEASE_TIPS.get(disease, "Consult an agronomist.")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if disease == "Healthy":
                st.success(f"### {emoji} {disease}")
            else:
                st.error(f"### {emoji} {disease}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.progress(int(confidence * 100))

        with col_r2:
            st.markdown(f"**💡 Recommended Action:**\n\n{tip}")
            st.markdown(f"**🌱 Crop:** {crop_type} | **🌍 Region:** {region} | **🌡️ Temp:** {temperature}°C")

        # Download
        out = input_df.copy()
        out["Predicted_Disease"] = disease
        out["Confidence_%"] = f"{confidence*100:.1f}"
        st.download_button(
            "⬇️ Download Classification Report (CSV)",
            data=out.to_csv(index=False).encode(),
            file_name="disease_classification_report.csv",
            mime="text/csv",
        )
