import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import shap
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ===============================
# Page config
# ===============================

st.set_page_config(
    page_title="Advanced CKM Risk Calculator",
    page_icon="🩺",
    layout="wide"
)


# ===============================
# Style
# ===============================

st.markdown(
    """
<style>

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.top-banner {
    background:#263c50;
    color:white;
    padding:30px;
    border-radius:10px;
    text-align:center;
    margin-bottom:25px;
}

.top-banner h1 {
    font-size:36px;
}

.result-card {
    border:1px solid #e5e7eb;
    border-radius:12px;
    padding:20px;
    background:white;
}

.note-box {
    background:#f7f8fb;
    padding:15px;
    border-radius:8px;
    color:#4b5563;
}

</style>
""",
    unsafe_allow_html=True
)



# ===============================
# Load model
# ===============================


@st.cache_resource
def load_model():

    with open(
        "xgb_model.pkl",
        "rb"
    ) as f:

        model = pickle.load(f)

    return model



try:

    model = load_model()


    # SHAP explainer
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional",
        model_output="probability"
    )


except Exception as e:

    st.error(
        f"Failed to load model: {e}"
    )

    st.stop()



# ===============================
# Features
# ===============================


if hasattr(
    model,
    "feature_names_in_"
):

    FEATURE_ORDER = list(
        model.feature_names_in_
    )

else:

    FEATURE_ORDER = [

        "DR",
        "lipid",
        "insulin",
        "age",
        "BUN",
        "UACR",
        "eGFR",
        "TG",
        "NEU",
        "Hb"

    ]



LABELS = {

    "DR":
    "Diabetic Retinopathy",

    "lipid":
    "Lipid-lowering Medication",

    "insulin":
    "Insulin Therapy",

    "age":
    "Age",

    "BUN":
    "Blood Urea Nitrogen",

    "UACR":
    "Urinary Albumin-Creatinine Ratio",

    "eGFR":
    "Estimated Glomerular Filtration Rate",

    "TG":
    "Triglycerides",

    "NEU":
    "Neutrophil Count",

    "Hb":
    "Hemoglobin"

}



DEFAULTS = {

    "DR":0,
    "lipid":0,
    "insulin":0,
    "age":60,
    "BUN":8,
    "UACR":200,
    "eGFR":70,
    "TG":1.6,
    "NEU":4,
    "Hb":120

}
# ===============================
# Header
# ===============================


st.markdown(
    """
<div class="top-banner">

<h1>
Advanced CKM Risk Identification<br>
in Patients with Diabetic Kidney Disease
</h1>

</div>
""",
    unsafe_allow_html=True
)



# ===============================
# Sidebar
# ===============================


st.sidebar.header(
    "Patient Information"
)


DR = (
    1
    if st.sidebar.selectbox(
        "Diabetic Retinopathy",
        ["No", "Yes"]
    ) == "Yes"
    else 0
)


lipid = (
    1
    if st.sidebar.selectbox(
        "Lipid-lowering Medication",
        ["No", "Yes"]
    ) == "Yes"
    else 0
)


insulin = (
    1
    if st.sidebar.selectbox(
        "Insulin Therapy",
        ["No", "Yes"]
    ) == "Yes"
    else 0
)



age = st.sidebar.number_input(
    "Age",
    value=int(DEFAULTS["age"])
)


BUN = st.sidebar.number_input(
    "Blood Urea Nitrogen",
    value=float(DEFAULTS["BUN"])
)


UACR = st.sidebar.number_input(
    "UACR",
    value=float(DEFAULTS["UACR"])
)


eGFR = st.sidebar.number_input(
    "eGFR",
    value=float(DEFAULTS["eGFR"])
)


TG = st.sidebar.number_input(
    "Triglycerides",
    value=float(DEFAULTS["TG"])
)


NEU = st.sidebar.number_input(
    "Neutrophil Count",
    value=float(DEFAULTS["NEU"])
)


Hb = st.sidebar.number_input(
    "Hemoglobin",
    value=float(DEFAULTS["Hb"])
)



threshold = st.sidebar.slider(
    "Risk threshold",
    0.05,
    0.95,
    0.5
)



calculate = st.sidebar.button(
    "Calculate Risk"
)



# ===============================
# Input dataframe
# ===============================


input_data = {

    "DR":DR,

    "lipid":lipid,

    "insulin":insulin,

    "age":age,

    "BUN":BUN,

    "UACR":UACR,

    "eGFR":eGFR,

    "TG":TG,

    "NEU":NEU,

    "Hb":Hb

}



X = pd.DataFrame(
    [
        {
            feature:
            input_data[feature]

            for feature in FEATURE_ORDER
        }
    ]
)



# ===============================
# Prediction
# ===============================


def predict_probability(
    model,
    data
):

    probability = (
        model.predict_proba(data)[0][1]
    )

    return probability



# ===============================
# SHAP function
# ===============================


def create_shap_plot(
    explainer,
    data
):


    shap_values = (
        explainer.shap_values(data)
    )


    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[1]



    if isinstance(
        explainer.expected_value,
        list
    ):

        base_value = (
            explainer.expected_value[1]
        )

    else:

        base_value = (
            explainer.expected_value
        )



    explanation = shap.Explanation(

        values=shap_values[0],

        base_values=base_value,

        data=data.iloc[0],

        feature_names=[
            LABELS.get(
                x,
                x
            )

            for x in data.columns
        ]

    )



    fig = plt.figure(
        figsize=(8,5)
    )



    shap.plots.waterfall(
        explanation,
        show=False
    )


    return fig
    # ===============================
# Gauge
# ===============================


def make_gauge(probability):

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability * 100,

            number={
                "suffix":"%"
            },

            title={
                "text":"Advanced CKM Risk"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":
                    "#ef476f"
                    if probability >= threshold
                    else "#2a9d8f"
                },

                "steps":[

                    {
                        "range":[0,50],
                        "color":"#d9f8e7"
                    },

                    {
                        "range":[50,100],
                        "color":"#ffe1ea"
                    }

                ]

            }

        )

    )


    fig.update_layout(
        height=350
    )


    return fig



# ===============================
# Main result
# ===============================


st.markdown(
    "## Prediction Result"
)



if calculate:


    try:

        probability = predict_probability(
            model,
            X
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.stop()



    st.markdown(

        f"""
<div class="result-card">

<h3>
Predicted probability of advanced CKM:
{probability*100:.1f}%
</h3>

</div>
""",

        unsafe_allow_html=True

    )



    st.plotly_chart(
        make_gauge(probability),
        use_container_width=True
    )



    # ===============================
    # Individual SHAP explanation
    # ===============================


    st.markdown("---")


    st.markdown(
        "## Individual-level SHAP-based Risk Explanation"
    )


    try:


        shap_fig = create_shap_plot(
            explainer,
            X
        )


        st.pyplot(
            shap_fig,
            use_container_width=True
        )


        st.caption(

            "The SHAP waterfall plot illustrates how each feature contributes to this individual's predicted probability of advanced CKM risk."

        )



    except Exception as e:


        st.warning(

            f"SHAP explanation unavailable: {e}"

        )



    st.markdown(

        """
<div class="note-box">

<b>Interpretation:</b><br>

Red bars indicate factors increasing predicted risk,
while blue bars indicate factors decreasing predicted risk.

<br><br>

This tool is intended for research demonstration
and should not replace clinical judgment.

</div>
""",

        unsafe_allow_html=True

    )



else:


    st.info(
        "Please enter patient information and click Calculate Risk."
    )



# ===============================
# Footer
# ===============================


st.markdown("---")


st.markdown(
    """
### Model Information

The XGBoost model uses the following predictors:

- Diabetic retinopathy
- Lipid-lowering medication
- Insulin therapy
- Age
- Blood urea nitrogen
- UACR
- eGFR
- Triglycerides
- Neutrophil count
- Hemoglobin

The online calculator provides individualized risk prediction
and SHAP-based interpretation of model output.
"""
)
