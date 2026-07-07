import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_PATH / "house_rent_pipeline.pkl"
DATA_PATH = BASE_PATH / "House_Rent_Dataset.csv"

st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #ff4b5c;
        --secondary: #6a31f6;
        --accent: #ffbe0b;
        --surface: #0f1117;
        --text: #e0e0e0;
        --card: rgba(30, 30, 46, 0.88);
    }

    body {
        background: radial-gradient(circle at top left, rgba(255, 75, 92, 0.08), transparent 25%),
                    radial-gradient(circle at bottom right, rgba(106, 49, 246, 0.06), transparent 30%),
                    linear-gradient(135deg, #0f1117 0%, #1a1a2e 45%, #16213e 100%);
        color: var(--text);
    }

    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .stApp {
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 1rem;
        backdrop-filter: blur(12px);
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border: none;
        color: white;
        font-weight: 700;
        box-shadow: 0 18px 45px rgba(255, 75, 92, 0.3);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 24px 55px rgba(255, 75, 92, 0.4);
    }

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div>div {
        border-radius: 18px;
        border: 1px solid rgba(106, 49, 246, 0.4);
        background: rgba(45, 45, 65, 0.95);
        color: #e0e0e0;
    }

    .hero-card,
    .detail-card,
    .summary-card {
        background: var(--card);
        border-radius: 28px;
        box-shadow: 0 22px 70px rgba(76, 10, 129, 0.25);
        border: 1px solid rgba(106, 49, 246, 0.3);
        padding: 1.5rem;
        animation: float 8s ease-in-out infinite;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(255, 75, 92, 0.15), rgba(106, 49, 246, 0.15));
    }

    .hero-title {
        font-size: 2.7rem;
        line-height: 1.05;
        margin: 0;
        color: #ffffff;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #b0b0c0;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    .section-heading {
        color: #bb86fc;
        font-weight: 800;
    }

    .metric-card {
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255, 190, 11, 0.15), rgba(255, 75, 92, 0.12));
        padding: 1rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_pipeline():
    if not PIPELINE_PATH.exists():
        st.error("Saved model file not found: house_rent_pipeline.pkl")
        st.stop()

    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Dataset not found: House_Rent_Dataset.csv")
        st.stop()

    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_option_lists(data: pd.DataFrame) -> dict:
    city_options = sorted(data["City"].dropna().unique().tolist())
    if "Lucknow" not in city_options:
        city_options.insert(0, "Lucknow")

    return {
        "Area Type": sorted(data["Area Type"].dropna().unique().tolist()),
        "City": city_options,
        "Furnishing Status": sorted(data["Furnishing Status"].dropna().unique().tolist()),
        "Tenant Preferred": sorted(data["Tenant Preferred"].dropna().unique().tolist()),
        "Point of Contact": sorted(data["Point of Contact"].dropna().unique().tolist()),
    }

pipeline = load_pipeline()
dataset = load_dataset()
options = get_option_lists(dataset)

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">House Rent Predictor</h1>
        <p class="hero-subtitle">Select property details, choose Lucknow or your favorite city, and get a vibrant rent estimate instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    left, right = st.columns((2.5, 1))

    with left:
        with st.form(key="rent_prediction_form"):
            st.markdown("## ✨ Enter home details")
            cols = st.columns(2)
            bhk = cols[0].selectbox("BHK", [1, 2, 3, 4, 5, 6], index=1)
            size = cols[1].number_input("Size (sqft)", min_value=100, max_value=10000, value=1200, step=50)

            cols = st.columns(2)
            area_type = cols[0].selectbox("Area Type", options.get("Area Type", ["Super Area", "Carpet Area", "Built-up Area"]))
            city = cols[1].selectbox("City", options.get("City", ["Lucknow", "Delhi", "Kolkata", "Mumbai", "Bangalore"]))

            cols = st.columns(2)
            furnishing_status = cols[0].selectbox(
                "Furnishing Status",
                options.get("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"]),
            )
            tenant_preferred = cols[1].selectbox(
                "Tenant Preferred",
                options.get("Tenant Preferred", ["Bachelors/Family", "Family", "Bachelors"]),
            )

            cols = st.columns(2)
            bathroom = cols[0].selectbox("Bathroom", [1, 2, 3, 4, 5], index=1)
            point_of_contact = cols[1].selectbox("Point of Contact", options.get("Point of Contact", ["Contact Owner", "Contact Agent"]))

            submit_button = st.form_submit_button("Predict Rent")

            if submit_button:
                input_df = pd.DataFrame(
                    [
                        {
                            "BHK": bhk,
                            "Size": size,
                            "Area Type": area_type,
                            "City": city,
                            "Furnishing Status": furnishing_status,
                            "Tenant Preferred": tenant_preferred,
                            "Bathroom": bathroom,
                            "Point of Contact": point_of_contact,
                        }
                    ]
                )

                prediction = pipeline.predict(input_df)
                predicted_rent = float(prediction[0])

                st.markdown("---")
                st.markdown(
                    f"<div class='summary-card'><h2>💰 Predicted Rent</h2><p style='font-size:2rem; font-weight:800; margin:0;'>₹{predicted_rent:,.0f}</p></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("## 📋 Input summary")
                st.table(input_df)

    with right:
        st.markdown("## 🎯 Model overview")
        st.markdown(
            "<div class='detail-card'><p>This app uses a pre-trained pipeline with encoding, scaling, and linear regression.</p></div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.markdown("<div class='metric-card'><strong>Dataset rows</strong><br>" + f"{len(dataset):,}</div>", unsafe_allow_html=True)
        col2.markdown("<div class='metric-card'><strong>Average rent</strong><br>" + f"₹{dataset['Rent'].mean():,.0f}</div>", unsafe_allow_html=True)

        st.markdown("### Available cities")
        st.write(", ".join(options["City"]))

        with st.expander("Dataset sample", expanded=False):
            st.write(dataset.head(8))

        with st.expander("Feature distributions", expanded=False):
            st.write(dataset[["BHK", "Size", "Bathroom", "Rent"]].describe())

        st.markdown(
            "#### Notes\n"
            "- Lucknow is included as a selectable city in this app.\n"
            "- Use the controls above to get an instant rent prediction.\n"
            "- The UI uses gradients, cards, and animated highlights for a premium feel."
        )

st.markdown("---")
st.caption("Built with Streamlit. Run: `streamlit run app.py`")

