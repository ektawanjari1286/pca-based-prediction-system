
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="ML Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
MODEL_PATH = Path(__file__).parent / "model (2).pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file nahi mili. "
        "model (2).pkl ko app.py ke same folder mein rakho."
    )
    st.stop()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()


# -------------------------------
# DASHBOARD HEADER
# -------------------------------
st.title("📊 Machine Learning Prediction Dashboard")
st.write(
    "Logistic Regression model ke through "
    "interactive classification predictions karo."
)

st.divider()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Dashboard Settings")
st.sidebar.success("Model loaded successfully")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Input features: 9")
st.sidebar.write("Classes: B and M")

if st.sidebar.button("🔄 Reset Inputs"):
    st.session_state.clear()
    st.rerun()


# -------------------------------
# INPUT FEATURES
# -------------------------------
st.subheader("📝 Enter Input Features")
st.caption(
    "PC1 se PC9 tak values enter karo. "
    "Training ke time use hui preprocessing aur scaling "
    "ke according values dena zaroori hai."
)

feature_names = [
    "PC1", "PC2", "PC3",
    "PC4", "PC5", "PC6",
    "PC7", "PC8", "PC9"
]

with st.form("prediction_form"):
    values = []

    # Display inputs in 3 columns
    columns = st.columns(3)

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            value = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=f"input_{feature}",
                help=f"Enter the trained model's {feature} value."
            )
            values.append(value)

    submitted = st.form_submit_button(
        "🚀 Predict Result",
        use_container_width=True
    )


# -------------------------------
# PREDICTION
# -------------------------------
if submitted:
    try:
        input_data = pd.DataFrame(
            [values],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📈 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Class",
                value=str(prediction)
            )

        with col2:
            st.metric(
                label="Model Type",
                value="Logistic Regression"
            )

        with col3:
            st.metric(
                label="Input Features",
                value=len(values)
            )

        # Show probability if supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in model.classes_],
                "Probability (%)": probabilities * 100
            })

            st.subheader("📊 Class Probabilities")

            chart_data = probability_df.set_index("Class")
            st.bar_chart(chart_data)

            st.dataframe(
                probability_df.style.format(
                    {"Probability (%)": "{:.2f}%"}
                ),
                use_container_width=True,
                hide_index=True
            )

        # Show input summary
        with st.expander("View Submitted Input Values"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.success("Prediction completed successfully!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption(
    "Built with Streamlit and Scikit-learn | "
    "Predictions depend on valid model inputs."
)